# Tests for variability calcs and outliers (zscore)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from raptor_parser import RaptorParser
import copy
import pandas as pd
import numpy as np

def test_calc_zscore():
    """ Compute group-wise zscore for kernels with the same name """
    mock_var_df = pd.DataFrame({
        'start':   [   0, 1000, 2000, 4000, 5000, 8000, 10000, 20000, 25000] ,
        'end':     [ 900, 2500, 2900, 5700, 6000, 9600, 17777, 22000, 29000] ,
        'gpuId':   [1,1,1,  1,1,1, 1, 1,1 ],
        'Kernel' : ['A', 'B', 'A', 'B', 'A', 'B', 'C', 'D', 'D']
        })
    var_raptor=RaptorParser(prekernel_seq=0, zscore_threshold=RaptorParser.default_zscore)
    var_raptor.set_op_df(copy.deepcopy(mock_var_df), set_roi=True)

    import math
    print(var_raptor.get_kernelseq_df())
    print ("op_df with Duration_zscore")
    op_df = var_raptor.get_op_df()
    print (op_df)

    dur_zscore = op_df['Duration_zscore']
    assert math.isclose(dur_zscore[1], -0.707, rel_tol=.001)
    assert math.isclose(dur_zscore[2], -1.225, rel_tol=.001)
    assert math.isclose(dur_zscore[3], -0.707, rel_tol=.001)
    assert math.isclose(dur_zscore[4], 1.225, rel_tol=.001)
    assert math.isclose(dur_zscore[5], 1.414, rel_tol=.001)
    assert math.isclose(dur_zscore[6], 0, rel_tol=.001)

    assert math.isclose(dur_zscore[7], 0, rel_tol=.001) # 'C" - only one kernel

    assert math.isclose(dur_zscore[8], -1, rel_tol=.001) # 'D' - two kernels
    assert math.isclose(dur_zscore[9], +1, rel_tol=.001) # 'D' - two kernels

def test_find_outliers():
    """ Flag outlier where zscore exceeds the specified threshold """

    # Create data-set where the 11th item is a black sheep outlier and make sure we can detect it
    mock_var_df = pd.DataFrame({
        'start':   [0]*11,
        'end':     [10000]*10 + [10],
        'gpuId':   [1]*11,
        'Kernel' : ['AAA']*11
        })
    var_raptor=RaptorParser(prekernel_seq=0, zscore_threshold=3)
    var_raptor.set_op_df(copy.deepcopy(mock_var_df), set_roi=True)
    print(var_raptor.get_kernelseq_df())
    print ("op_df with Duration_zscore")
    op_df = var_raptor.get_op_df()
    print (op_df)
    for i in range(10):
        assert op_df.iloc[i].Outlier == False
    assert op_df.iloc[10].Outlier 
