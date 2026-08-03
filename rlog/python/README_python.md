# rlog Python Client

Python client for the rlog logging hub. Loads `librlog.so` via ctypes and
provides the same core API as the C++ client in `include/rlog/client.h`.

## Requirements

- `librlog.so` must be built and installed (or available via `LD_LIBRARY_PATH`)

## Quick Start

```python
from rlog import RlogClient

client = RlogClient()

# Guard expensive logging with the cached flag
if client.is_logging:
    client.range_push("my_api", "x=1")
    # ... work ...
    client.range_pop()
```

## API Reference

### Constructor

```python
client = RlogClient(lib_path="librlog.so")
```

Loads the rlog hub library from `lib_path`. Registers an internal callback
that keeps `is_logging` in sync with the hub's active state.

### is_logging

```python
if client.is_logging:
    client.mark("op", "data=42")
```

A cached `bool` that tracks whether any logging tool is attached. Updated
automatically via a hub callback. Use this to guard logging calls and avoid
dispatch overhead when no tool is listening (same pattern as the C++ guard
benchmark).

### mark

```python
client.mark(apiname, args, domain=None, category=None)
```

Emit a single marker event. When `domain` or `category` are omitted, the
defaults set by `set_default_domain` / `set_default_category` are used.

```python
# All four arguments
client.mark("allocate", "size=1024", domain="MyApp", category="memory")

# Using defaults
client.set_default_domain("MyApp")
client.set_default_category("memory")
client.mark("allocate", "size=1024")
```

### range_push / range_pop

```python
client.range_push(apiname, args, domain=None, category=None)
# ... work ...
client.range_pop()
```

Push and pop a named range. Ranges can be nested. `domain` and `category`
default the same way as `mark`.

### is_active

```python
active = client.is_active()
```

Live query to the hub. Returns `True` when at least one logger is registered.
Prefer checking `is_logging` in hot paths since it avoids the function call
into the shared library.

### register_active_callback

```python
def on_change():
    print("active:", client.is_active())

client.register_active_callback(on_change)
```

Register an additional callback that fires when the hub's active state
changes (logger added or removed). The built-in callback that updates
`is_logging` is always registered; use this for application-specific
reactions.

### get_property

```python
value = client.get_property(domain, property, default_value)
```

Look up a configuration property from the hub's property store.

```python
timeout = client.get_property("MyApp", "request_timeout", "30")
```

### set_default_domain / set_default_category

```python
client.set_default_domain("MyApp")
client.set_default_category("network")
```

Set defaults used by `mark` and `range_push` when `domain` or `category`
are not passed explicitly. Defaults start as empty strings.

## Complete Example

```python
from rlog import RlogClient

client = RlogClient()
client.set_default_domain("MyApp")
client.set_default_category("compute")

def run_kernel(n):
    if client.is_logging:
        client.range_push("run_kernel", f"n={n}")

    # ... do work ...

    if client.is_logging:
        client.range_pop()

for i in range(1000):
    run_kernel(i)
```

## Installation

The Python client is installed by CMake alongside the C++ headers and hub
library:

```
cmake --build build --target install
```

This places `rlog.py` in `/usr/local/lib/python/rlog/`. Add that path to
`PYTHONPATH` or copy the file into your project.
