# Tests for multi-GPU transition points and filtering

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
    assert np.isnan(op_df.iloc[0].PreGap_ns)
    assert op_df.iloc[1].PreGap_ns == 1.0
    assert op_df.iloc[2].PreGap_ns == 4.0

    assert np.isnan(op_df.iloc[3].PreGap_ns) # should reset to NAN for first record in new GPU
    assert op_df.iloc[4].PreGap_ns == 1.0
    assert op_df.iloc[5].PreGap_ns == 0.0

    assert list(op_df['Duration_ns'].T) == [9,6,5, 3,14,1]
    assert list(op_df['sequenceId'].T)  == [1,2,3, 1,2,3]

    # make sure we can print it,
    raptor.print_op_trace()

def test_gpu_filter():
    r = copy.deepcopy(raptor)
    
    assert r.sql_filter_str() == "where start>=0 and start<=25"
    r.set_gpu_id(1)
    assert r.sql_filter_str() == "where start>=0 and start<=25 and gpuId==1"

def test_gpu_df():
    gpu_df = raptor.get_gpu_ts_df()
    gpu_df = raptor.get_gpu_ts_df(duration_unit='ns')
    print(gpu_df)
    assert gpu_df['Idle_pct'].iloc[0] == 20
