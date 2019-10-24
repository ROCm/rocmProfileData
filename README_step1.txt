Run you application under rocprofiler.  You will need a working rocprofiler that logs all required fields. (harder than it sounds)

Use: 
echo pmc: > in.txt
rocprof -i in.txt --hip-trace --timestamp on -d rocout

This should populate a folder tree that will terminate in a dir containing:
hcc_ops_trace.txt
hip_api_trace.txt

Those are the goodies you need for step 2.
