Nvidia Visual Profiler files can be converted to RPD with nvvp2rpd.py

Concerns:
  Nvvp files do not have Api names but rather store enum values.  Those values can be looked up in 'cupti_runtime_cbid.h', which is a generated file and may change at any time.
  The nvvp2rpd.py importer requires a 'cupti_runtime_cbid.h'.
  The safest approach would be to aquire that 'cupti_runtime_cbid.h' from the system being profiled

Current Status:
The importer is at a very early statge of development (and may stay there).  It imports all api calls.  It imports kernel ops (only names and durations).  Many other ops, e.g. memcpy, memset, sync, are not imported.  There is detailed data for all these things is the source files.

Usage:
python3.6 nvvp2rpd.py --help

