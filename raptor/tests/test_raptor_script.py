import os

raptor = "./raptor.py "

(test_path, test_file) = os.path.split(__file__)
rpd_file = os.path.join(test_path, "mytrace.rpd.gz")

def test_help():
    assert not os.system(raptor + " --help")

def test_trace():
    assert not os.system(raptor + rpd_file + " -c -t")

def test_zscore():
    assert not os.system(raptor + rpd_file + " -ct -z 3")

def test_instance():
    assert not os.system(raptor + rpd_file + " -i 0 -z 3 --op-trace-cmd-width=60")
