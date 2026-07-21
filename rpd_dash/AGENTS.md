# AGENTS.md — rpd_dash Development Guide

This file helps AI coding assistants work effectively in this Dash-based project.
Read this before writing or modifying any code.

## Project Structure

```
rpd_dash/
  rpd_dash/
    app.py              # App factory, layout, sidebar, NAV_LINKS, Flask routes
    pages/              # One file per page — auto-discovered by Dash
    util/               # Shared Python modules (db.py, html_table.py, etc.)
    assets/             # Static JS/CSS — auto-served by Dash, no registration needed
  setup.py              # Package definition — add dependencies to install_requires
  RPD_INFO.md           # Schema reference consumed by the chat feature
```

### Adding a new page

1. Create `rpd_dash/pages/my_page.py`
2. Call `dash.register_page(__name__, path="/my-page", name="My Page")` at module level
3. Define a `layout()` **function** (not a variable) that returns a component tree
4. Add `("My Page", "/my-page")` to `NAV_LINKS` in `app.py`
5. Run `make install` — the app runs from installed paths, not source

### Database access

All pages access the SQLite database through `rpd_dash.util.db`:
- `db.rpd_path` — the loaded file path (None if no file loaded)
- `db.get_connection()` — fresh connection, caller must close
- `db.query_df(sql)` — returns a pandas DataFrame, handles connection lifecycle
- Guard every `layout()` with: `if not db.rpd_path: return html.Div("No RPD file loaded.")`

### Build and Testing

Always run `make install` after any code change. The app runs from the installed
package in `/opt/venv`, not from the source tree. The `build/` directory is a stale
artifact — never edit files there.

**Do not test page modules with direct import.** Running
`python -c "from rpd_dash.pages import chat"` will always fail with
`validate_use_pages` error — this is expected. Dash page modules can only be
imported by the Dash app's page discovery mechanism. To test, run the full app:
`rpd-viewer <file.rpd>` or `make install && rpd-viewer`.

---

## Dash Callback Rules

### CRITICAL: Each Output property may belong to exactly ONE callback — PERIOD

**THIS IS THE MOST COMMON CRASH. THE APP WILL NOT START. NOT EVEN CLOSE. IT WILL DIE AT LAUNCH.**

If two `@callback` decorators both list
`Output("my-div", "children")`, Dash raises a **startup error** and the app
will not load at all. **Period. End of story.**

**Wrong — two callbacks share an output:**
```python
@callback(Output("messages", "children"), Input("send", "n_clicks"), ...)
def send_message(...): ...

@callback(Output("messages", "children"), Input("clear", "n_clicks"), ...)  # CRASH
def clear_messages(...): ...
```

**Right — one callback handles both interactions:**
```python
@callback(
    Output("messages", "children"),
    Input("send", "n_clicks"),
    Input("clear", "n_clicks"),
    ...,
    prevent_initial_call=True,
)
def handle_messages(send_clicks, clear_clicks, ...):
    if ctx.triggered_id == "clear":
        return []
    # handle send...
```

If you need to initialize a component's display from stored data, do it in
`layout()`, not in a separate callback.

### Use `prevent_initial_call=True` on interactive callbacks

Without it, the callback fires on page load with None/0 inputs. This causes
crashes when the callback body assumes real user input. All interactive callbacks
in this project use `prevent_initial_call=True`.

### Use `dash.no_update` to skip outputs selectively

When a callback has multiple outputs but you only want to update some of them,
return `dash.no_update` for the others. Do not return `None` — that clears
the component.

### Use `ctx.triggered_id` to distinguish which Input fired

```python
from dash import ctx

@callback(Output(...), Input("btn-a", "n_clicks"), Input("btn-b", "n_clicks"), ...)
def handle(a_clicks, b_clicks, ...):
    if ctx.triggered_id == "btn-a":
        ...
    elif ctx.triggered_id == "btn-b":
        ...
```

### Use `@callback`, not `@app.callback`

This project uses the module-level `callback` decorator imported from `dash`.
Do not use `app.callback` — the `app` object is not accessible from page modules.

```python
from dash import callback, Input, Output, State
```

---

## Dash Component Pitfalls

### `dcc.Store`

- `storage_type` controls persistence:
  - `"memory"` (default) — lost on page refresh
  - `"session"` — survives refresh, lost when tab closes
  - `"local"` — survives across sessions (localStorage)
- A Store's `data` property is both an Input and an Output — be careful not to
  create a second callback that writes to the same Store.

### `dcc.Markdown`

Use `dcc.Markdown(text)` to render markdown including tables, code blocks, bold,
and lists. Do **not** hand-roll a markdown parser or import the `markdown` library
just to feed HTML into `html.Div`. The `dcc.Markdown` component handles GFM
(GitHub Flavored Markdown) natively.

`dangerously_allow_html` is a property of `dcc.Markdown`, **not** `html.Div`.

### `dcc.Loading`

Wrap any component whose content is updated by a slow callback:
```python
dcc.Loading(
    id="loading-wrapper",
    type="circle",
    children=html.Div(id="slow-content"),
)
```
The spinner appears automatically while the callback updating `slow-content`
is running. No JavaScript or Interval polling needed.

### `dcc.Input` vs `dcc.Textarea`

- `dcc.Input` has `n_submit` (fires on Enter key) — use it for single-line inputs.
- `dcc.Textarea` does **not** have `n_submit` or `n_clicks`. To handle Enter-to-send
  on a Textarea, use a JavaScript file in `assets/` that listens for keydown and
  clicks the send button programmatically.

### `dcc.Interval`

Do not use `dcc.Interval` to simulate async behavior or polling for callback
results. Dash callbacks are synchronous — the loading state is handled by
`dcc.Loading`. Using Interval to poll creates split-callback architectures that
violate the one-output rule. **If you need long-running async work, do it in
the callback body (e.g., with threads) and use `dcc.Loading` for the spinner.**

---

## Clientside Callbacks

`dash.clientside_callback()` is a **function call**, not a decorator.

**Wrong:**
```python
@dash.clientside_callback(  # CRASH — returns None, Python tries to use None as decorator
    "function(...) { ... }",
    Output(...), Input(...),
)
def placeholder():
    pass
```

**Right:**
```python
dash.clientside_callback(
    "function(...) { ... }",
    Output(...), Input(...),
)
# No def — it's a plain function call, not a decorator
```

### JavaScript context

Inside clientside callback JS code, the Python `dash` module does not exist.
Use `window.dash_clientside.no_update` instead of `dash.no_update`.

For keyboard event handling and other DOM interactions, prefer placing a
standalone `.js` file in the `assets/` directory. Dash auto-serves all files
in `assets/` — no registration or import needed.

---

## Styling Conventions

This app has a **dark sidebar** and a **light content area**. They use different
color palettes — do not mix them.

**Sidebar** (defined in `app.py`):
- Background: `#2c2c2c`
- Text / nav links: `#ddd`
- Muted text: `#888`
- Borders: `#444`
- Nav links use a shared `link_style` dict with `color: "#ddd"`

**Content area** (the page body, right of sidebar):
- Background: white (default, no override)
- Card/surface background: `#f5f5f5`
- Primary text: default (black/dark) — no color override needed
- Muted/label text: `#888`
- Tables use `make_table()` from `util/html_table.py` with alternating `#fafafa` rows

New pages should follow the light content area style. Use inline styles consistent
with `dashboard.py` and `_card()`. Do not apply dark backgrounds (`#1e1e1e`, `#2c2c2c`)
to the content area — that is only for the sidebar.

---

## Common Patterns in This Codebase

### Page with no callbacks (most pages)
```python
import dash
from dash import html
from rpd_dash.util import db

dash.register_page(__name__, path="/example", name="Example")

def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")
    conn = db.get_connection()
    try:
        # query and build components...
        return html.Div([...])
    finally:
        conn.close()
```

### Page with interactive callbacks
```python
import dash
from dash import html, dcc, callback, Input, Output, State, ctx

dash.register_page(__name__, path="/interactive", name="Interactive")

def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")
    return html.Div([
        # components with ids...
    ])

@callback(
    Output("result", "children"),
    Input("action-a", "n_clicks"),
    Input("action-b", "n_clicks"),
    State("some-input", "value"),
    prevent_initial_call=True,
)
def handle_actions(a_clicks, b_clicks, value):
    if ctx.triggered_id == "action-a":
        ...
    elif ctx.triggered_id == "action-b":
        ...
```

### Read-only SQL execution
```python
conn = sqlite3.connect(f"file:{db.rpd_path}?mode=ro", uri=True, timeout=10)
conn.execute("PRAGMA query_only = ON")
try:
    cur = conn.execute(sql)
    ...
finally:
    conn.close()
```
