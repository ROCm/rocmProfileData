# rpd_dash — RPD Viewer

A web-based viewer for `.rpd` trace files produced by
[rocmProfileData](https://github.com/ROCm/rocmProfileData). It's a
[Dash](https://dash.plotly.com/) app: point it at a trace file, open the
URL it prints in your browser, and explore GPU kernels, API calls, timelines,
counters, and more — no need to load huge JSON traces into Perfetto/Chrome
just to answer a quick question.

## Install

```bash
cd rpd_dash
make install       # pip install .
```

This installs the `rpd-viewer` console script into your active Python
environment.

## Running it

```bash
rpd-viewer /path/to/trace.rpd
```

Then open `http://localhost:8050` in a browser.

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `rpd_file` (positional) | none | Path to the `.rpd` file to load. Optional — if omitted, the app starts with a "Load RPD File" screen where you can enter a path in the browser. |
| `--host` | `0.0.0.0` | Interface to bind to. |
| `--port` | `8050` | Port to bind to. |
| `--no-debug` | off | Disable Dash debug mode (auto-reload, error overlay). |

If the trace file's modification time is recent (< 30s), the Dashboard page
assumes it's a **live** profiling session still being written to and
auto-refreshes stats via SSE.

## Pages

The sidebar groups pages into a main nav and an "Analysis" section.

- **Dashboard** (`/`) — landing page with summary stats, GPU busy time, and
  a domain breakdown for the loaded trace.
- **Kernels** (`/kernel`) — kernel launch summary: call counts, total/average
  duration, ranked by GPU time.
- **API Calls** (`/api`) — CPU-side API call summary (hip, torch, miopen, etc.).
- **GPU Ops** (`/op`) — raw GPU-side operation summary.
- **Copies** (`/copy`) — memory copy operations (H2D/D2H/D2D), sizes and
  bandwidth.
- **Timeline** (`/trace`) — generates Chrome Tracing JSON from the trace and
  opens it directly in [Perfetto](https://ui.perfetto.dev/), or lets you
  download the JSON.
- **GPU Monitor** (`/monitor`) — time-series charts of hardware counters
  (power, temperature, clocks, etc.) if present in the trace.
- **Graphs** (`/graphs`) — HIP graph captures: launches, kernels within each
  graph, and timing.
- **Autograd** (`/autograd`) — PyTorch autograd operator → kernel breakdown,
  for correlating backward-pass ops with GPU kernels.
- **Metadata** (`/metadata`) — raw key/value metadata stored in the trace
  (schema version, pid/gpu striding, etc.).
- **Counters** (`/counters`) — GPU hardware counter values per kernel.
- **SQL Query** (`/query`) — a free-form, read-only SQL query console against
  the trace's SQLite database, with example queries to get started.
- **Analysis / Timeline views** (`/tl/...`) — focused timeline breakdowns:
  GPU Timeline, Kernel Categories, Short Kernels, Torch Ops, Ops by Category.
- **Chat** (`/chat`) — see below.

## Chat page

The Chat page (`/chat`) lets you ask natural-language questions about the
loaded trace. The assistant is given a read-only `run_sql` tool and a schema
reference (`RPD_INFO.md`) describing the trace's tables/views, and it iterates
running SQL queries against the trace database to build its answer.

Chat talks to any **OpenAI-compatible** chat-completions endpoint (vLLM,
llama.cpp server, Ollama, text-generation-webui, actual OpenAI, etc.) via the
`openai` Python client. **The backend model must support tool/function
calling** — without it, the assistant can't query the trace.

Configure the connection with environment variables before starting
`rpd-viewer`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RPD_CHAT_BASE_URL` | `http://localhost:8000/v1` | Base URL of the OpenAI-compatible server. |
| `RPD_CHAT_MODEL` | `default-model` | Model name to request from the server. |
| `RPD_CHAT_API_KEY` | `not-needed` | API key/token, if the server requires one. |
| `RPD_CHAT_MAX_TURNS` | `20` | Max tool-call round trips per question before giving up. |
| `RPD_CHAT_TIMEOUT` | `120` | Overall chat timeout, in seconds. |

Example, pointing at a local [vLLM](https://github.com/vllm-project/vllm)
server:

```bash
export RPD_CHAT_BASE_URL="http://localhost:8000/v1"
export RPD_CHAT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
rpd-viewer trace.rpd
```

Example, pointing at OpenAI directly:

```bash
export RPD_CHAT_BASE_URL="https://api.openai.com/v1"
export RPD_CHAT_MODEL="gpt-4o"
export RPD_CHAT_API_KEY="sk-..."
rpd-viewer trace.rpd
```

## Developing

See [`AGENTS.md`](AGENTS.md) for the internal project structure, page/callback
conventions, and styling rules used by this Dash app.
