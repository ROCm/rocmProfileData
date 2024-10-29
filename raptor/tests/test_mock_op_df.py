# Tests using a mock op_df table 

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from raptor_parser import RaptorParser
import pandas as pd
import numpy as np
custom_ops_df = pd.DataFrame({
    'start':[0,10,20, 2,6, 8], 
    'end'  :[9,16,25, 5,20,9],
    'gpuId':[0,0,0,   1,1,1],
    'Kernel' : ['A', 'B', 'C', 'AA', 'B', 'C']
    })

raptor=RaptorParser()
raptor.set_op_df(custom_ops_df)

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
