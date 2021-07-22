This is a tracer that can attach to any process and record hip apis, ops, and roctx.

Steps:
1) in this directory run 'make'.  It should build rpd_tracer.so
2) cd ../rocpd_python and 'python setup.py install'  Installs rpd utilites
3) Run 'runTracer.sh -o <output_file>.rpd <your_command_and_args>
4) 'python ../rpd2tracing <your_profile>.rpd <your_profile>.json' for chrome tracing output

WARNING: keep runTracer.sh and rpd_trace.so together in $PWD.  (until a proper installer exists)

ISSUE: Very small traces will not write any output. This a tmp workaround. Profile something bigger.


Manual Stuff:
 - Use 'LD_PRELOAD=./rpd_tracer.so' to attach the profiler to any process
 - Default output file name is 'trace.rpd'
 - Override file name with env 'RPDT_FILENAME='
 - Create empty rpd file with python3 -m rocpd.schema --create ${OUTPUT_FILE}
 - Multiple processes can log to the same file concurrently
 - Files can be appended any number of times
