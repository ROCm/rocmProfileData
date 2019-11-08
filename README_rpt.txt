Output from 'HIP_TRACE_API=1' and 'HCC_PROFILE=2' can be imported into RPD with rpt2rpd.py.

Support is minimal:
  Captures all api calls
  Captures kernel ops (only names and durations)
  Other ops are not imported 
  Api -> op relationship is not available in the source data
  Api duration is not available in the source data (currently set to 10 usec for fun)

Todo:
  Capture all ops
  Generate ops 'subclasses' to capture more detailed op data.  Some is available from the source.
  Attempt to link api to ops based on the assumption of in-order queues

Usage:
  python3.6 rpt2rpd.py --help
