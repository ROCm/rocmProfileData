This is a module to add ranged and "impulse" user markers directly from python.

Usage examples:

    from hipScopedMarker import hipScopedMarker

    hipScopedMarker.emitMarker("This is a single event with no duration")
    with hipScopedMarker("This is a range"):
        pass	# this block's duration is logged

Notes:
    The current rocprofiler backend only logs push and pop events.  You have to assemble these calls into ranges if you are using the RocTX logs directly.  The rocprof2rpd.py importer does this automatically.


Installation:

This requires python3.x.
This incarnation requires rocm2.9+.

CC=/opt/rocm/bin/hipcc python3.6 setup.py install
