# Rlog

[![CI](https://github.com/ROCm/rlog/actions/workflows/ci.yml/badge.svg)](https://github.com/ROCm/rlog/actions/workflows/ci.yml)

--------------------------------------------------------------------------------

Rlog is an API for passing annotation messages between applications and profilers.
It is designed to be lightweight, composable, and zero-cost when no tool is listening.

Contents:
<!-- toc -->

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Client](#client)
- [Tool](#tool)
- [Properties](#properties)
- [rlog-config](#rlog-config)

<!-- tocstop -->

---

## About

The rlog system has three roles:

- **Applications** — programs or libraries that want to expose profiling annotations
  such as named ranges, markers, and domain-scoped events.
- **Tools** — profilers, loggers, or other recording systems that listen for and
  process annotations in real time.
- **Hub** — a shared library (`librlog.so`) that routes annotation calls from
  applications to all currently registered tools.

The key design property is decoupling: an application compiles against the small
client API (`rlog.cpp` + `rlog/client.h`) and has no compile-time dependency on
any specific profiling tool. The hub is loaded at runtime via `dlopen`. If no hub
is present, all logging calls are no-ops. If a hub is present but no tool has
registered, the `isActive()` call returns false and the application can skip
annotation work entirely.

Multiple tools can be registered simultaneously. Each tool receives every
annotation event dispatched by every application in the process.

Applications can also store and retrieve logging configuration as named properties
in a persistent database, scoped by domain. Default values are registered
automatically on first access and can be tuned at any time using `rlog-config`.

### Supported backends

In addition to the native rlog protocol, the hub can forward annotations to
legacy GPU profiling APIs:

| Backend | Library | Environment variable to override path |
|---------|---------|---------------------------------------|
| rlog (native) | `librlog.so` | — |
| ROCm ROCTx | `libroctx64.so` | `RLOG_ROCTX_LIBPATH` |
| NVIDIA NVTX | `libcupti.so` | `RLOG_NVTX_LIBPATH` |

Legacy backends can be force-enabled regardless of tool registration state
using `RLOG_FORCE_ROCTX=1` or `RLOG_FORCE_NVTX=1`. This is useful when using
profilers that are not rlog-aware.

---

## Features

### Logging configuration via properties

Applications can store their logging configuration — such as which domains or
categories to enable, verbosity levels, or output destinations — as named
properties in the persistent database. Properties are scoped by domain, making
it natural to store each application's settings under its own domain name.

For example, an application might read its configuration at startup:

```cpp
const char* verbose  = rlog::getProperty("my_application", "verbose",   "0");
const char* maxDepth = rlog::getProperty("my_application", "max_depth", "10");
```

If no entry exists, the default value is written to the database automatically.
This means the first run of an application self-registers its configuration keys
with their defaults, which can then be inspected and tuned using `rlog-config`
without modifying source code.

This replaces the common pattern of scattering configuration across hard-to-discover
environment variables such as `NCCL_DEBUG` or `MIOPEN_LOG_LEVEL` that must be set
before each run, are not visible at runtime, and have no central place to inspect
or document them.

---

## Installation

Dependencies:
- CMake 3.5 or later
- A C++14-capable compiler
- `libsqlite3-dev` (for the property database and `rlog-config`)

Build and install:

```sh
mkdir build
cd build
cmake ..
make
make install
```

This installs:
- `/usr/local/lib/librlog.so` — the hub shared library
- `/usr/local/bin/rlog-config` — the property inspection tool
- `/usr/local/include/rlog/client.h` — client API header
- `/usr/local/include/rlog/Logger.h` — tool interface header
- `/usr/local/include/rlog/Hub.h` — hub declaration (for tools)

To override the install prefix:

```sh
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/rlog
```

---

## Client

A client is any application or library that wants to emit annotations.

### Setup

Clients must:
- Include `rlog/client.h`
- Compile and link `src/rlog.cpp`

`rlog::init()` must be called once before any other rlog functions and before
concurrent logging begins. It reads environment variables, resolves library
paths, and loads the hub. It is not thread-safe.

### Active callback

Register a callback to be notified when a tool starts or stops recording.
The callback should update a flag that guards annotation work, avoiding
overhead when no tool is listening.

```cpp
namespace rlog {
    bool isLogging = false;

    void onActiveChanged() {
        isLogging = rlog::isActive();
    }

    class Client {
    public:
        Client() {
            rlog::init();
            rlog::registerActiveCallback(&onActiveChanged);
            rlog::setDefaultDomain("my_application");
            rlog::setDefaultCategory("");
        }
    };

    Client client;
} // namespace rlog
```

`setDefaultDomain` and `setDefaultCategory` set fallback values used by the
two- and one-argument overloads of `mark` and `rangePush`. They must be called
before concurrent logging begins and are not thread-safe.

### Logging ranges and markers

```cpp
void myFunction(const char* input) {
    if (rlog::isLogging) {
        rlog::rangePush("myFunction", input);
    }

    // ... do work ...

    if (rlog::isLogging) {
        rlog::rangePop();
    }
}

void myEvent() {
    if (rlog::isLogging) {
        rlog::mark("myEvent", "fired");
    }
}
```

All three overloads are available:

```cpp
// Full form: explicit domain and category
rlog::rangePush("my_application", "io", "readFile", filename);

// Omit domain: uses setDefaultDomain value
rlog::rangePush("io", "readFile", filename);

// Omit domain and category: uses both defaults
rlog::rangePush("readFile", filename);
```

### Reading properties

Applications can read configuration values from the persistent property store:

```cpp
const char* timeout = rlog::getProperty("my_application", "timeout", "30");
```

If no entry exists for `("my_application", "timeout")`, it is created with
the value `"30"` and `"30"` is returned. On subsequent calls the stored value
is returned, even if a different default is supplied.

### Environment variables

| Variable | Effect |
|----------|--------|
| `RLOG_ROCTX_LIBPATH` | Override path to the ROCTx shared library |
| `RLOG_NVTX_LIBPATH` | Override path to the NVTX shared library |
| `RLOG_FORCE_ROCTX=1` | Force ROCTx logging on regardless of tool registration |
| `RLOG_FORCE_NVTX=1` | Force NVTX logging on regardless of tool registration |

---

## Tool

A tool is a profiler, logger, or any system that wants to receive rlog annotations.

### Setup

Tools must:
- Implement the interface defined in `rlog/Logger.h`
- Include `rlog/Hub.h`
- Link against `librlog.so`

### Logger interface

```cpp
// rlog/Logger.h
namespace rlog {
    class Logger {
    public:
        virtual ~Logger() = default;
        virtual void mark(const char* domain, const char* category,
                          const char* apiname, const char* args) = 0;
        virtual void rangePush(const char* domain, const char* category,
                               const char* apiname, const char* args) = 0;
        virtual void rangePop() = 0;
    };
}
```

### Registering and unregistering

```cpp
#include "rlog/Hub.h"

class MyTool : public rlog::Logger {
public:
    void startRecording() {
        rlog::Hub::singleton().addLogger(*this);
    }

    void stopRecording() {
        rlog::Hub::singleton().removeLogger(*this);
    }

    void mark(const char* domain, const char* category,
              const char* apiname, const char* args) override {
        // record the event
    }

    void rangePush(const char* domain, const char* category,
                   const char* apiname, const char* args) override {
        // record range start
    }

    void rangePop() override {
        // record range end
    }
};
```

`addLogger` and `removeLogger` are reference-counted per logger instance.
Calling `addLogger` twice on the same logger requires two `removeLogger` calls
to fully unregister it. When the last logger is removed, all registered
active callbacks are invoked with `isActive()` returning false.

---

## Properties

The property system provides persistent, domain-scoped key-value storage shared
across all processes that use the hub.

Properties are stored in `$HOME/.rlog.db` (a SQLite3 database). The schema is:

```sql
CREATE TABLE properties (
    domain   TEXT NOT NULL,
    property TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (domain, property)
);
```

### Behavior

- All values are strings.
- If a property does not exist when `getProperty` is called, it is inserted with
  the supplied default value. The default value is then returned.
- Subsequent calls return the stored value, even if a different default is passed.
- Property values set via `rlog-config set` take effect immediately for any
  process that calls `getProperty` after the write (cache aside — within a
  single process, the first read is cached for the process lifetime).

### Thread and process safety

- Multiple threads within a process are serialized through the Hub mutex.
- Multiple processes are safe via SQLite WAL mode and `BEGIN IMMEDIATE` transactions.

---

## rlog-config

`rlog-config` is a command-line tool for inspecting and editing the property
database at `$HOME/.rlog.db`.

### List all properties

```sh
rlog-config
```

Output is grouped by domain, one domain per section:

```
[my_application]
  retries = 3
  timeout = 30

[other_tool]
  debug = false
  verbosity = 1
```

### Get a property value

```sh
rlog-config get <domain>:<property>
```

Prints the current value to stdout. Exits with a non-zero status if the
property does not exist.

```sh
$ rlog-config get my_application:timeout
30
```

### Set a property value

```sh
rlog-config set <domain>:<property> <value>
```

Creates the property if it does not exist, or updates it if it does.
The database file is created if it does not yet exist.

```sh
$ rlog-config set my_application:timeout 60
$ rlog-config get my_application:timeout
60
```
