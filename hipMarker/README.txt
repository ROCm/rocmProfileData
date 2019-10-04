This is a module to add ranged and "impulse" user markers directly from python.

Usage examples:

    import hipMarker
    from hipScopedMarker import hipScopedMarker

    hipMarker.emitMarker("This is a single event with no duration")
    with hipScopedMarker("This is a range"):
        pass	# this block's duration is logged

Notes:
    The current backend only supports instant events, so the hipScopedMarker emits two name-mangled events.  You can demangle these.  The rocprof2rpd.py importer does this automatically.


Installation:

Acquire a current internal rocprofiler tarball.  There is no install location for this stuff so we will hardcode one.  Extract the package into: '/data/Downloads/rocprofiler_pkg'  Edit setup.py if you need a different location.

CC=/opt/rocm/bin/hipcc python3.6 setup.py install
