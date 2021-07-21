Use rocprof2rpd.py to convert the rocprofiler file to an rpd file:
'python3.6 rocprof2rpd.py --ops_input_file hcc_ops_trace.txt --api_input_file hip_api_trace.txt myCoolProfile.rpd'

To use the python library rocpd_python, navigate to the rocpd_python folder and install the library

python3.6 setup.py install
python3.6 -m rocpd.rocprofiler_import --ops_input_file hcc_ops_trace.txt --api_input_file hip_api_trace.txt myCoolProfile.rpd


Generate json from your profile:
'python3.6 rpd2tracing.py myCoolProfile.rpd trace.json'

Good to go, fire up chrome and take a look.

Note:
If you need to subrange you must specify timestamps in usecs.  You can browse the the db to find good start/end times but those times are in nsec.  Divide by 1000.
----------------------------------------------------------------------------
usage: rpd2tracing.py [-h] [--start START] [--end END] input_rpd output_json

convert RPD to json for chrome tracing

positional arguments:
  input_rpd      input rpd db
  output_json    chrone tracing json output

optional arguments:
  -h, --help     show this help message and exit
  --start START  start timestamp
  --end END      end timestamp
-----------------------------------------------------------------------------

Bonus:
There are some views in HelpfulQueries.txt.  You can open your rpd file with sqlite3 and apply those views.
- 'Op' and 'api' mirror the the 'rocpd_op' and 'rocpd_api' tables but have the strings joined in.  
- 'Top' gives a top kernel summary
- 'Busy' gives gpu busy time
