# devdocs — internal developer field guide

Developer documentation for the rocmProfileData codebase: architecture, code
layout, invariants, and procedures for making changes. Skill-style progressive
loading: read this router, then open only the files your task needs.

> Not to be confused with `skills/` — that is a user-facing, distributable agent
> skill for *using* the tools. This folder documents the *codebase itself* and
> never ships.

## Task router

| I want to... | Read |
|---|---|
| get the big picture (first session here) | [architecture.md](architecture.md) |
| build from a fresh clone, add a dependency, fix CI | [build.md](build.md) |
| work on the tracer library (rpd_tracer/) | [subsystems/rpd-tracer.md](subsystems/rpd-tracer.md) |
| add or modify a DataSource | [recipes/add-datasource.md](recipes/add-datasource.md) + [concepts/datasources.md](concepts/datasources.md) |
| add a config/env-var knob | [recipes/add-config-flag.md](recipes/add-config-flag.md) |
| add/change a schema table or view | [recipes/add-schema-table.md](recipes/add-schema-table.md) + [concepts/rpd-schema.md](concepts/rpd-schema.md) |
| work on rlog (annotations library) | [subsystems/rlog.md](subsystems/rlog.md) |
| work on rocm-trace-lite (RTL) | [subsystems/rocm-trace-lite.md](subsystems/rocm-trace-lite.md) |
| work on multi-node / distributed tracing | [subsystems/remote-and-distributed.md](subsystems/remote-and-distributed.md) + [concepts/timestamps-and-clocksync.md](concepts/timestamps-and-clocksync.md) |
| work on the web viewer | [subsystems/viewer.md](subsystems/viewer.md) (also `rpd_viewer/AGENTS.md`) |
| work on the `rocpd` python package or tools/ scripts | [subsystems/python-and-tools.md](subsystems/python-and-tools.md) |
| understand how the tracer gets into a process | [concepts/injection-and-interposition.md](concepts/injection-and-interposition.md) |
| touch timestamps, clocks, correlation ids | [concepts/timestamps-and-clocksync.md](concepts/timestamps-and-clocksync.md) |
| touch threading, buffers, flush, shutdown, init order | [concepts/threads-and-shutdown.md](concepts/threads-and-shutdown.md) |
| before shipping: known traps, dead code, stale docs | [gotchas.md](gotchas.md) |

## Ground rules for these files

- Every doc is self-contained but ≤ ~200 lines; link out rather than duplicate.
- Point *at* code (`path/file.cpp:line`) instead of restating logic — line
  numbers drift, so re-verify the symbol still lives there when it matters.
- Existing canonical docs are linked, never copied:
  - `rpd_viewer/RPD_INFO.md` — schema-as-analysis reference (also the chat
    agent's system prompt; loaded by `rpd_viewer/rpd_viewer/chat_api/__init__.py:22-24`)
  - `rpd_viewer/AGENTS.md` — viewer-contributor guide
  - `docs/README_tracer_quickstart.md`, `docs/README_embedded.md`,
    `docs/README_distributed.md`, `docs/README_clocksync.md`,
    `docs/README_gpucounters.md`, `docs/README_torchrun.md`
  - `rlog/README.md` (owned by the `rlog` branch), `rocm-trace-lite/docs/`
  - `docs/perf_playbooks/` — analysis workflows
- Authoritative schema DDL is `rocpd_python/rocpd/schema_data/*.cmd` — everything
  else is a copy or a view of it.
