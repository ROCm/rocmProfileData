import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import pytest
from raptor_parser import RaptorParser
import pandas as pd
import numpy as np

(test_path, test_file) = os.path.split(__file__)
rpd_file = os.path.join(test_path, "mytrace.rpd.gz")

def test_make_roi_plus():
    raptor = RaptorParser(rpd_file, roi_start="+10ms", roi_end="+16.7ms")
    raptor.print_timestamps()
    assert raptor.roi_start_ns == 10*1e6
    assert raptor.roi_end_ns == 16.7*1e6

def test_make_roi_minus():
    raptor = RaptorParser(rpd_file, roi_start="-8ms", roi_end="-5ms")
    raptor.print_timestamps()
    assert raptor.roi_start_ns == 490594451
    assert raptor.roi_end_ns ==   493594451

def test_make_roi_pct():
    raptor = RaptorParser(rpd_file, roi_start="50%", roi_end="70%")
    raptor.print_timestamps()
    assert raptor.roi_start_ns == 249297225
    assert raptor.roi_end_ns ==   349016115 

def test_make_roi_from_kernel():
    raptor = RaptorParser(rpd_file)
    raptor.print_timestamps()
    assert raptor.roi_start_ns == 0
    raptor.set_roi_from_str(roi_start="Cijk_")
    print("\nAfter setting ROI to kernel name:")
    raptor.print_timestamps()
    assert raptor.roi_start_ns == 10496683

def test_make_roi_from_bad_kernel():
    raptor = RaptorParser(rpd_file)
    with pytest.raises(RuntimeError):
        raptor.set_roi_from_str(roi_start="ZZZ_")

def test_empty_roi():
    """ Ensure code can handle ranges with no ops """
    raptor = RaptorParser(rpd_file, roi_start="97%", roi_end="98%")
    raptor.print_timestamps()
    print(raptor.get_op_df())
    print(raptor.get_kernelseq_df())
    print(raptor.get_category_df())
    print(raptor.get_variability_df())
