# New Features

--------------------------------------------------------------------------------

This is a changelog that describes new features as they are added.  Newest first.


Contents:
<!-- toc -->

- [Rpd_tracer Start/stop](#rpd_tracer-startstop)
- [Schema v2](#schema-v2)

<!-- tocstop -->


--------------------------------------------------------------------------------

## Rpd_tracer Start/stop
Rpd_tracer recording is able to be toggled by the program being traced.  This allows for fine grained control over what is collected.
This is useful for:
- "trimming" files at collection time by simply not recording the warmup phase
- capturing managable amounts of data from long-running programs

#### From Python
There is an rpdTracerControl class available (as part of 'make install').  This class is able to load the rpd_tracer shared library, create an rpd file, and toggle recording.  It can be used as a context manager to do scoped recording, or to start/stop recording directly.

```
from rpdTracerControl import rpdTracerControl

# Optionally call this class method before creating first instance
rpdTracerControl.setFilename(name = "trace.rpd", append=False)

# Create first instance (this loads the profiler and creates the file)
profile = rpdTracerControl()

profile.start()
# recorded section
profile.stop()

with rpdTracerControl() as p:
  pass # recorded section

```

**Notes**:

The calls to start/stop recording are ref counted.  It is safe to use multiple recursive instances to record individual sections.  Do consider however, recording will remain active if any instance has it set on.

There are special considerations when using with python multiprocessing.  From the main process, before spawning any other processes:
  - Set the filename and append mode
  - Create one instance of the rpdTracerControl class

    *(This initializes an empty trace file (or appends existing) and sets up ENV variables used by the child processes.)*



#### From C/C++
This is possible but very much manual:
- create an empty rpd file (via python tools)  *See runTracer.sh for an example*
- suppress autostart of recording (via ENV)
- optionally set filename (via ENV)
- dlopen() librpd_tracer.so
- see "rpd_tracer.h" for symbols

Relevant ENV variables:
- RPDT_FILENAME = "./trace.rpd" 
- RPDT_AUTOSTART = 0



--------------------------------------------------------------------------------

## Schema V2
The RPD schema was changed to support copy and kernel args better. Previously kernel info and copy info were linked to an "op", i.e. gpu activity. Now they are linked to the api call that caused the execution.  This is required so that syncronous copies (ones that do not generate gpu ops) have a place to store args.

The new tables are:
- *rocpd_copyapi*
- *rocpd_kernelapi*

Both tables have a '*api_ptr_id*' column that can be used to join them to the api call (*rocpd_api*)  

The api call (*rocpd_api*) can be joined to the gpu op (*rocpd_op*) through the correlation id table (*rocpd_api_ops*)

The following views will be available in newly created files:
- **kernel**  
    Kernel info, with gpu timing
- **copy**  
    All copies, with cpu timing (i.e. the timespan of api call)
- **copyop**  
    Subset of copies that generated gpu ops, with gpu timing

There is now a key-value pair in (*rocpd_metadata*) with the schema version. E.g. ("schema_version", "2")
