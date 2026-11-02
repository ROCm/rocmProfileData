# rpd_viewer — the web viewer

Dash/Plotly Python app (not Go, not Django) over one `.rpd` file. Console
script `rpd-viewer` (`rpd_viewer/setup.py:21-25`; deps dash, dash-ag-grid,
plotly, pandas, openai, rocpd). **`rpd_viewer/AGENTS.md` is the authoritative
contributor guide — read it before touching the app.** This file is the
orientation layer.

## Lineage (so old references don't mislead you)

Root-level **Django** app (manage.py + rocpd/views.py) → deleted `3819abb`;
only remnant is `rocpd_python/rocpd/util/chrometracing.py`. Then **rpd_dash/**
(Dash) added `ac7381a` → deleted & renamed to **rpd_viewer/** in `0d2144b`.
Today `rpd_dash/` on disk is untracked build litter; git tracks nothing there.

## Run / build

- `cd rpd_viewer && make install` (plain `pip install .`), then
  `rpd-viewer trace.rpd [--host 0.0.0.0] [--port 8050] [--no-debug]`
  (argparse `rpd_viewer/rpd_viewer/app.py:16-22`).
- Root CMake pip-installs it as an ALL target after rocpd
  (`CMakeLists.txt:106-120`) — disable with `-DRPD_BUILD_VIEWER=OFF`.
- **Always run the installed app; `build/` here is a stale artifact — never
  edit it** (AGENTS.md). Don't import page modules directly to test them.

## Layout & extension points

- `app.py` — app factory (`use_pages=True` auto-discovery, `:35-40`),
  `NAV_LINKS` (`:42-56`), Flask JSON/HTML routes (`:231-266`), SSE live
  endpoints (`:268-306` — "live" = file mtime < 30s), `/tracedata`
  (Perfetto JSON via `rocpd.tracing.generate_rpd_json`, RO connection
  `:311-316`), `/download-rpd`.
- `pages/` — ~21 pages (Dashboard, Kernels, API, Ops, Copies, Timeline,
  Monitor, Graphs, Autograd, Metadata, Counters, SQL query, `/tl/*`, Chat).
  Adding a page = add module + NAV_LINKS entry; follow existing page style.
- `util/db.py` — the DB access layer: module-global path (`:8`),
  `get_connection()` (`:21-26`, WAL pragma — note this can create side files
  on a non-WAL db), `query_df()` with SHA256 result cache (`:78-94`),
  `query_df_indexed()` materializing `tmp_api`/`tmp_api_ops` (`:40-51`),
  feature probes `table_exists/column_exists/has_annotations/has_torch_ops`
  (`:110-150`) — **the viewer detects capability by column/table presence,
  never by the `schema_version` tag.**
- `chat_api/__init__.py` — LLM chat blueprint at `/api/chat`: any
  OpenAI-compatible tool-calling backend (`RPD_CHAT_BASE_URL` default
  `http://localhost:8000/v1`, `RPD_CHAT_MODEL`, `RPD_CHAT_API_KEY`,
  `RPD_CHAT_MAX_TURNS=20`, `RPD_CHAT_TIMEOUT=120`); single `run_sql` tool
  on a read-only connection (`mode=ro` + `PRAGMA query_only`, `:44-48`),
  500-row / 6000-char caps; **system prompt is `RPD_INFO.md` loaded verbatim
  (`:22-24`)** — schema-doc edits directly change chat behavior.

## House rules (from AGENTS.md, enforced in review)

- Read-only SQL execution for anything user-submitted (chat, `/query`,
  `/tracedata` all use `mode=ro`).
- Guard optional features behind the `util/db.py` probes so v1/v2 and
  counter-less traces degrade gracefully instead of erroring.
- `RPD_INFO.md` doubles as product surface (chat prompt) and doc — keep it
  trace-agnostic; `RPD_RESEARCH.md` is its regeneration prompt.
