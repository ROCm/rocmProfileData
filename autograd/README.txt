This patch allow the pytorch autograd profiler to log directly to the rpd format.

PATCH
-----
To apply the patch (to an installation or src respectively):
cd <path>/site-packages    (e.g. ~/.local/lib/python3.6/site-packages/)
  or
cd <src-root>              (e.g. ~/pytorch/)
git apply rpd_profile.patch


USAGE
-----
Once installed use the profiler like normal.  There is an additional output option to output rpd.
Example:

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
       <code to profile>
    prof.export_rpd("tracefile.rpd")
