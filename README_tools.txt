Tools Hidden in Plain Sight
---------------------------

Pytorch Autograd:
Patch pytorch allowing autograd to write directly to rpt.

HipMarker (poorly named):
Insert user markers (instrumentation) into your python code.  Logs to roctx markers and ranges.

TopEx:
SQL to output top kernel and api summary (exclusive times).  Inclusive time is available via a view embedded by most rpt importers.
example: 'sqlite3 myprofile.rpt < topEx.cmd'


