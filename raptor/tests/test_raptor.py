import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from raptor_parser import RaptorParser
import pandas as pd
import numpy as np

df = pd.DataFrame([[1000,2000, 132], [2000,5000, 500], [5000,7000, 1200]], columns=['start','end','value'])

(test_path, test_file) = os.path.split(__file__)
rpd_file = os.path.join(test_path, "test.rpd.gz")
print (f"{__file__=}, {rpd_file=}")
raptor = RaptorParser(rpd_file)

def test_make_roi_plus():
    raptor = RaptorParser(rpd_file, roi_start="+10ms", roi_end="+16.7ms")
    raptor.print_timestamps()
    assert raptor.roi_start_ns == 10*1e6
    assert raptor.roi_end_ns == 16.7*1e6

def test_make_roi_minus():
    raptor = RaptorParser(rpd_file, roi_start="-8ms", roi_end="-5ms")
    raptor.print_timestamps()
    assert raptor.roi_start_ns == 3489389688550768
    assert raptor.roi_end_ns ==   3489389691550768

def test_make_roi_pct():
    raptor = RaptorParser(rpd_file, roi_start="50%", roi_end="70%")
    raptor.print_timestamps()
    assert raptor.roi_start_ns == 3489389447253542
    assert raptor.roi_end_ns ==   3489389546972432 
    
def test_gaps_label():
    assert RaptorParser._make_gaps_labels(
        (0,np.inf)) == ["GAP >0us"]
    assert RaptorParser._make_gaps_labels(
        (0,20,np.inf)) == ["GAP <20us", "GAP >20us"]
    assert RaptorParser._make_gaps_labels(
        (0,20,100,np.inf)) == ["GAP <20us", 
                               "GAP 20us-100us",
                               "GAP >100us"]
