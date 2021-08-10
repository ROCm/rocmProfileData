This directory contains scripts and tools to assist with analyzing traces. 
Each tool will include usage and docs within the body.

Sqlite3 scripts
---------------
Files ending in .cmd are raw sql scripts.  They can be executed simalar to:
    sqlite3 trace_file.rpd < script.cmd

These scripts can do just about anything, including:
  - complex reports
  - adding views (which will persits in the .rpd)
  - trimming or modifying existing data
  - creating new tables to hold supplmental data

Python scripts
--------------
Some tasks are better suited for python, which can also directly modify the rpd files.
Check the individual usage by running with no args.  General format:
    python script.py <args>
