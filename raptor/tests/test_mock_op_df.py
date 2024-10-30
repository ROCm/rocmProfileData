# Tests using a mock op_df table 

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from raptor_parser import RaptorParser
import copy
import pandas as pd
import numpy as np
custom_op_df = pd.DataFrame({
    'start':[0,10,20, 2,6, 8], 
    'end'  :[9,16,25, 5,20,9],
    'gpuId':[0,0,0,   1,1,1],
    'Kernel' : ['A', 'B', 'C', 'AA', 'B', 'C']
    })

raptor=RaptorParser()
raptor.set_op_df(custom_op_df, set_roi=True)

def test_multi_gpu():

    op_df = raptor.get_op_df()
    print(op_df)
    assert np.isnan(op_df.iloc[0].PreGap)
    assert op_df.iloc[1].PreGap == 1.0
    assert op_df.iloc[2].PreGap == 4.0

    assert np.isnan(op_df.iloc[3].PreGap) # should reset to NAN for first record in new GPU
    assert op_df.iloc[4].PreGap == 1.0
    assert op_df.iloc[5].PreGap == 0.0

    assert list(op_df['Duration_ns'].T) == [9,6,5, 3,14,1]
    assert list(op_df['sequenceId'].T)  == [1,2,3, 1,2,3]

    # make sure we can print it,
    raptor.print_op_trace()

def test_gpu_filter():
    r = copy.deepcopy(raptor)
    
    assert r.sql_filter_str() == "where start>=0 and start<=25"
    r.set_gpu_id(1)
    assert r.sql_filter_str() == "where start>=0 and start<=25 and gpuId==1"


def test_calc_zscore():
    mock_var_df = pd.DataFrame({
        'start':   [   0, 1000, 2000, 4000, 5000, 8000, 10000] ,
        'end':     [ 900, 2500, 2900, 5700, 6000, 9600, 17777] ,
        'gpuId':   [1,1,1,  1,1,1, 1 ],
        'Kernel' : ['A', 'B', 'A', 'B', 'A', 'B', 'C']
        })
    var_raptor=RaptorParser(prekernel_seq=0)
    var_raptor.set_op_df(copy.deepcopy(mock_var_df), set_roi=True)

    import math
    print(var_raptor.get_top_df())
    print ("op_df with Duration_zscore")
    op_df = var_raptor.get_op_df()
    print (op_df)

    dur_zscore = op_df['Duration_zscore']
    assert math.isclose(dur_zscore[0], -0.707, rel_tol=.001)
    assert math.isclose(dur_zscore[1], -1.225, rel_tol=.001)
    assert math.isclose(dur_zscore[2], -0.707, rel_tol=.001)
    assert math.isclose(dur_zscore[3], 1.225, rel_tol=.001)
    assert math.isclose(dur_zscore[4], 1.414, rel_tol=.001)
    assert math.isclose(dur_zscore[5], 0, rel_tol=.001)

def test_find_outliers():
    mock_var_df = pd.DataFrame({
        'start':   [0]*11,
        'end':     [10000]*10 + [10],
        'gpuId':   [1]*11,
        'Kernel' : ['AAA']*11
        })
    var_raptor=RaptorParser(prekernel_seq=0, zscore_threshold=3)
    var_raptor.set_op_df(copy.deepcopy(mock_var_df), set_roi=True)
    print(var_raptor.get_top_df())
    print ("op_df with Duration_zscore")
    op_df = var_raptor.get_op_df()
    print (op_df)
    for i in range(10):
        assert op_df.iloc[i].Outlier == False
    assert op_df.iloc[10].Outlier 
