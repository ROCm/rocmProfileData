# RAPTOR
## Introduction
Tool for parsing and post-processing RPD files:
- slice, dice, and eviscerate RPD files into focused regions-of-interest.
- generate summaries of top kernels and combine into categories (ie GEMM, Comm, Attention, etc).
- compute “Gaps” where GPU is idle.
- compute kernel to kernel variability and percentage of execution time
- auto-roi feature to focus on hottest region
- tables in the RaptorParser class are organized into Pandas dataframes for interactive analysis via ipython/jupyter/etc.
- show a text trace of each command's execution - kernel name, duration, idle gaps, etc
- RAPTOR possibly stands for ROCm Profile-Data Tool and Output Refinement.

## Quickstart and Usage
### Command-line interface
From inside the ./raptor/ directory (or add the full path to the script)
```
# Print usage:
./raptor.py --help
```

```
# Print category and top kernel tables:
./raptor.py tests/mytrace.rpd.gz --categorize --top
```

```
# Print category and top kernel tables, eliminating time before&after the hottest kernel first&last execution
./raptor.py tests/mytrace.rpd.gz --categorize --top --auto-roi
```

```
./raptor.py tests/mytrace.rpd.gz --op-trace
```

### Interactive Python
The RaptorParser class can be used directly from interactive python (ipython on Linux, Jupyter, or VSCode).
This provides an interactive experience which can be useful to iteratively explore the data - examine some results, then dive deeper on areas of interest, all within a Python experience.

For example: get a list of the top kernels, then list all the instances of the top kernel and sort by kernel duration. Or - 

The RaptorParser class can also be included into other scripts - for example to extract the top5 kernels for a list of RPD files.
See the cookbook/ directory for some specific examples.

## Insider Info

### Kernel sequences
Raptor 'top' kernels are uniqified using a sequence of preceding kernels.
This is useful to disambiguate cases where the same kernel is called in different contexts and thus aggregating on kernel name
conflates information from multiple contexts.
Sequences are useful in particular for automatic variability detection.
They are less useful in cases with high degrees of dynamic behavior where the tensor sizes change even for the same sequence of kernels.
Set --prekernel-seq to set the number of kernels to use.
"0" will disable the sequence feature and aggregate on just the kernel name.

### Region-of-Interest (ROI)
Sometimes RPD trace collection precisely captures the desired hot ROI, and this is the preferred flow when the user is familiar with the application under test and able to modify the source to enable/disable the collection.
In cases where this not possible, raptor provides tools to explicitly or automatically set the ROI.
The ROI is applied to all subsequent commands (category, tables, op-trace).
In interactive mode, the ROI can be changed, and this will force re-computation of the tables.

#### Explicit ROI
--roi-start, --roi-end parms explicity specify the start and end ROI.
The first time in the trace is always 0.    Start timestamps must be <= end timestamps.

#### ROI Units
The default unit is 'ms', but ns, us, s, or sec may also be used. These all specify 123.45ms:
   - `--roi-start 123.45`
   - `--roi-start 123.45ms`
   - `--roi-start 123450ns`
   - `--roi-start .12345s`

#### Start- and End- Relative 
Use a leading plus sign to specify the timestamp is relative to the trace start (always 0, so this is same as no leading +)
   - `--roi-start=+123.45`

Use a leading minus sign to indicate timestmaps is negative relative to the trace end.  Note use of the = sign so the shell can parse the negative sign correctly.
This example sets ROI to a 90ms region near the end of the trace:

   - `--roi-start=-100ms --roi-end=-10ms`

#### ROI based on kernel name regex
Specify a regular expression that matches one or more kernel names.  The ROI will be set to the first occurence of the
matching kernel name.
   - `--roi-start=Cijk_`

### Categories

Raptor combines kernels into higher-level categories for a better view of the "forest".  For example: 

```
Categories:
           UniqKernels  TotalCalls  TotalDur_ms  VarSum_ms  AvgDur_us  Pct
GEMM                11      164673       3574.0      210.8       21.7 42.0
_COMM               10       87722       2028.6      255.3       23.1 23.8
_Other              28      292485       1745.2      443.2        6.0 20.5
Attention            2       81920        706.6       69.3        8.6  8.3
_GPU_Idle            6      136769        317.4        0.0        2.3  3.7
aten                25        9903        142.2       10.2       14.4  1.7
```

#### Builtin Categories
Raptor has several builtin category definitions which start with a leading underscore:
- _COMM : Communication kernels such as AllReduce or AllGather.  These include a barrier synchronization.
Raptor treats COMM kernels special for variability calculations.
- _GPU_Idle : GPU is idle
- _Variability : Attempt to categorize the gap between the fastest instance of a specific kernel vs the actual implementation.
- _Other : Kernels which don't match any defined category.  

#### Custom Categories
Categories are specified in a dictionary-like JSON file with the name of the category and corresponing list of kernel name regular expressions.
The top view includes a "Category" column.  Kernels which don't match any defined category are shown as "_Other".

The default file is show as an example below.  Users can add new categories (or expand the regex for existing ones) to get more clarity on "_Other" kernels.
This can be useful if the top list shows a large percentage of kernels in the "_Other" category.
If a kernel matches multiple categories, the LAST category in the file wins - so add more specific categories at the END of the file.
```
$ cat raptor_cat_vllm.json
{
    "GEMM" : ["^Cijk_"],
    "CopyD2D" : ["CopyDeviceToDevice"],
    "aten" : ["^.* at::"],
    "_COMM" : ["ncclDevKernel_Generic","void vllm::cross_device_reduce"],
    "topk" : ["^.*topk"],
    "Attention" : ["paged_attention*", "attn_fwd"]
}
```

### Gaps and GPU-Idle
Raptor computes the time before each kernel when the GPU is idle - this can be useful to detect underutilization due to exposed 
host activity or synchronization or other causes.
Gaps are shown in the top_df summary ("PreGap"), and in each record of the op-trace.
In the category summary, gaps are accumlated into a "_GPU_Idle" field.

### Variability
TODO

## Reference
### Directory Structure
- raptor_parser.py : Class for reading and parsing RPD files.
- raptor.py : Command-line driver.
- ./cookbook : Interactive python examples and recipes showing how to use the RaptorParser API.
- ./tests : PyTest-format tests and test inputs. 

### Testing and Contribution Guidelines

   - Ensure all tests pass Green before submitting.
   - Add new tests for new features, following the format of the existing tests.
   - To run the tests:
```
$ pytest tests/
```

### TODO
- Dataframes for analysis