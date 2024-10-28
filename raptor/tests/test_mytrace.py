import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
# Read an actual RPD file and call all the key functions (top, categories, op_trace, etc):

import pytest
from raptor_parser import RaptorParser
import pandas as pd
import numpy as np

(test_path, test_file) = os.path.split(__file__)
rpd_file = os.path.join(test_path, "mytrace.rpd.gz")
raptor = RaptorParser(rpd_file)

def test_print_op_trace():
    print(raptor.get_op_df())
    raptor.print_op_trace(max_ops=50)

def test_top_df():
    print(raptor.get_top_df())

def test_pretty_top_df():
    print(raptor.get_pretty_top_df())

def test_cat_df():
    print(raptor.get_category_df())
