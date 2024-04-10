# Rocm Profile Data

--------------------------------------------------------------------------------

ROCm Profile Data is a collection of tools for tracing and analyzing gpu related activity on a system.  This is represented by a timeline of api calls, app log messages, async gpu operations, and related interactions/dependencies.


Contents:
<!-- toc -->

- [About](#about)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Tools](#tools)
  - [runTracer.sh](#runtracer.sh)
  - [sqlite3](#sqlite3)
  - [rpd2tracing.py](#rpd2tracing.py)

<!-- tocstop -->


## About

The ROCm Profiler Data or RPD ecosystem consists of the following:
- Collection tools - Profilers capable of detecting and logging gpu activity
- File format - A standard file format (based on sqlite3) for collection and analysis tools to work with
- Analysis tools - Programs capable of interpreting the profile data in a meaningful way.  These can be written in SQL, C/C++, python, etc.


## Installation

RPD makes heavy use to SQLite(https://sqlite.org/index.html).  You will want to install the runtime and dev packages before preceeding.  E.g.
```
apt-get install sqlite3 libsqlite3-dev
```

Additional packages required
```
apt-get install libfmt-dev
```

There are many tools in the RPD repo.  A baseline set can be built and installed via make:
```
make; make install
```
This will install python modules that are used to manipulate trace files.
It will also build and install the native tracer, rpd_tracer.


## Quickstart

+ Install per the [Installation](#installation) section.
+ Try each of the [Tools](#tools) below, in order.


## Tools

#### runTracer.sh
RunTracer.sh is used to launch and trace a program.  It is installed in the system path as part of 'make install'.  It can trace any process and its subprocesses.
```
runTracer.sh python exampleWorkload.py
```
By default the profile will be written to "trace.rpd".

#### sqlite3
Quick inspection of trace data can be performed with the sqlite3 command line
```
sqlite3 trace.rpd

sqlite> select count(*) from rocpd_api;
978621
sqlite> select count(*) from rocpd_op;
111899
sqlite> select * from top;
Name                                      TotalCalls  TotalDuration  Ave         Percentage
----------------------------------------  ----------  -------------  ----------  ----------------
Cijk_Alik_Bljk_SB_MT64x128x16_SN_1LDSB0_  3180        3670897        1154        33.1596545434822
Cijk_Alik_Bljk_SB_MT64x128x16_MI32x32x2x  12720       1703806        133         15.3906835540479
Cijk_Alik_Bljk_SB_MT128x128x16_MI32x32x1  3180        1471672        462         13.2937917318908
void at::native::legacy::elementwise_ker  22525       1059802        47          9.57331814908329
void at::native::legacy::elementwise_ker  13593       515243         37          4.65425092430873
...
sqlite> .exit

```

#### rpd2tracing.py
Trace data can be viewed in chrome://tracing for visualization.  Rpd2tracing.py exports the trace contents as a json file suitable for chrome://tracing or perfetto.
```
python3 tools/rpd2tracing.py trace.rpd trace.json
```

