# Recipe: add a config / env-var knob

The tracer has one config resolution path — use it, don't add `getenv`.

## 1. Choose a name pair

Env `RPDT_<UPPER_SNAKE>` + property key `<lower_snake>`, both in the
`rpd_tracer` property domain (rlog `$HOME/.rlog.db`). Precedence is fixed:
`rpd_setConfig()` > env var > rlog property > default
(`rpd_tracer/Config.cpp:23-33`). Known exceptions you should NOT copy:
clocksync/logagg/SMI knobs are env-only raw `getenv` (legacy; FIXME
`rpd_tracer/Firefly.h:29`).

## 2. Read it where it belongs

In the consuming DataSource or Logger:
`getConfig("RPDT_MYFOO_X", "myfoo_x", 3)` — call at init/configure time, not
per-event (string compares + property lookup are not hot-path cheap).
For a kill-switch knob consumed before any instance exists, follow
`Logger.cpp:141-198` (list knobs) or `Utility.h` getters.

## 3. Document it in the four places it will be looked up

- the embedded API property table: `docs/README_embedded.md:29-41`
- the quickstart env table: `docs/README_tracer_quickstart.md:115-137`
- `rpd_capi.cpp:52-63` (inline list maintained beside `rpd_setConfig`)
- [subsystems/rpd-tracer.md](../subsystems/rpd-tracer.md) env table here

## 4. Wire adjacent surfaces if it changes output

- Viewer: gate features with `util/db.py` probes, not with config reads.
- If the knob is something users must set at launch, consider a
  `runTracer.sh` flag that sets it (script reads persistent defaults via
  `rlog-config`, `runTracer.sh:23-26` — mirror that if it should persist).
- If it belongs to rlog itself (backend selection, forcing), do it in the
  `rlog` branch (`RLOG_*` pattern, `rlog/src/rlog.cpp:112-134`), then bump
  the submodule.

## 5. Verify all three tiers

```
RPDT_MYFOO_X=1 runTracer.sh ...                    # env
rlog-config set rpd_tracer:myfoo_x 1 && runTracer.sh ...   # property
```
and confirm precedence by setting env != property.
