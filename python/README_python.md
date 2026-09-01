# rlog Python Client

Python client for the rlog logging hub. Loads `librlog.so` via ctypes and
provides the same core API as the C++ client in `include/rlog/client.h`.

## Requirements

- `librlog.so` must be built and installed (or available via `LD_LIBRARY_PATH`)

## Quick Start

```python
from rlog import RlogClient

client = RlogClient()

if client.is_logging:          # guard: build no arg strings when idle
    client.range_push("my_api", "x=1")
# ... work happens regardless ...
if client.is_logging:          # sample again; never reuse the value above
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

Unlike the C++ client, Python cannot make an idle range free: budget roughly
150 ns per range (guarded raw calls or the decorator) and ~430 ns for the
context manager, paid whether or not a tool is attached. Instrument work
measured in microseconds, not inner loops. See `../OPTIMIZATIONS.md`.

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

Guard both calls with `is_logging`, sampling it separately each time. Do not
cache the push-time value and reuse it at pop: if tracing resumes mid-range the
pop must still be delivered, or every range opened after the resume is reported
at the wrong nesting depth.

### range / range_decorator

```python
with client.range("my_api", lambda: f"x={expensive()}"):
    ...                                    # work

@client.range_decorator(args=lambda n: f"n={n}")
def my_api(n):
    ...                                    # apiname defaults to the func name
```

Scope helpers that push, pop and guard correctly, including the two-sample rule
above. `args` may be a plain string, or a callable that is only invoked while
logging is active — use the callable form whenever building the string costs
anything, since a plain argument is evaluated before the range is entered.

Prefer the decorator: its arguments are the wrapped function's own, so nothing
extra is computed. The context manager allocates an object on every entry.

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
