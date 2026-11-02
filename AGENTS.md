# AGENTS.md

Start with **`devdocs/README.md`** — the internal developer field guide
(architecture, build, subsystems, concepts, recipes, gotchas) with a task
router for "I want to change X → read Y".

Scope notes:
- `skills/` is a user-facing distributable skill for *using* the tools —
  unrelated to `devdocs/`, do not merge concepts between them.
- `rpd_viewer/AGENTS.md` owns viewer-specific conventions.
- Contributions to `rlog/` target the `rlog` branch (submodule of this repo).
- Schema edits go in `rocpd_python/rocpd/schema_data/*.cmd`, never in
  generated `rpd_tracer/*Schema.h`.
- Build with CMake (`cmake -B build -S . && cmake --build build`); the
  per-directory `rpd_tracer/Makefile` is stale and produces a different
  library. See `devdocs/build.md`.
