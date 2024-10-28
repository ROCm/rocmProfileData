# Test misc functions and utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from raptor_parser import RaptorParser
import numpy as np


def test_gaps_label():
    assert RaptorParser._make_gaps_labels(
        (0,np.inf)) == ["GAP >0us"]
    assert RaptorParser._make_gaps_labels(
        (0,20,np.inf)) == ["GAP <=20us", "GAP >20us"]
    assert RaptorParser._make_gaps_labels(
        (0,20,100,np.inf)) == ["GAP <=20us", 
                               "GAP 20us-100us",
                               "GAP >100us"]
