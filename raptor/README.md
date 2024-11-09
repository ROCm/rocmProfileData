# RAPTOR
## Introduction
Tool for parsing and post-processing RPD files:
- slice, dice, and eviscerate RPD files into focused regions-of-interest.
- generate summaries of kernels and combine into categories (ie GEMM, Comm, Attention, etc).
- compute “Gaps” where GPU is idle.
- compute kernel to kernel variability and percentage of execution time
- auto-roi-top feature to focus ROI on the hottest kernel
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
# Print category and kernel tables:
./raptor.py tests/mytrace.rpd.gz --categorize --kernelseq
```

```
# Print category and kernelseq tables, eliminating time before&after the hottest kernel first&last execution
./raptor.py tests/mytrace.rpd.gz --categorize --kernelseq --auto-roi-top
```

```
./raptor.py tests/mytrace.rpd.gz --op-trace
```

### Interactive Python
The RaptorParser class can be used directly from interactive python (ipython on Linux, Jupyter, or VSCode).
This provides an interactive experience which can be useful to iteratively explore the data - examine some results, then dive deeper on areas of interest, all within a Python experience.

For example: get a list of the hot kernels, then list all the instances of the kernel and sort by kernel duration. Or - 

The RaptorParser class can also be included into other scripts - for example to extract the top5 kernels for a list of RPD files.
See the cookbook/ directory for some specific examples.

- Important Dataframes 
   - op_df
   - kernelseq_df
   - category_df
   - variability_df
   - pretty_kernelseq_df
- Caching
- Using the API from a script

## Insider Info

### Kernel sequences
Raptor kernels are uniqified using a sequence of preceding kernels.
This is useful to disambiguate cases where the same kernel is called in different contexts and thus aggregating on kernel name
conflates information from multiple contexts.
Sequences are useful in particular for automatic variability detection.
They are less useful in cases with high degrees of dynamic behavior where the tensor sizes change even for the same sequence of kernels.
Set --prekernel-seq to set the number of kernels to use.
"0" will disable the sequence feature and aggregate on just the kernel name.

### Region-of-Interest (ROI)
Sometimes RPD trace collection precisely captures the desired hot ROI, and this is the preferred flow when the user is familiar with the application under test and able to modify the source to enable/disable the collection.
In cases where this not possible, raptor provides tools to explicitly or automatically set the ROI.
The ROI is applied to all subsequent commands (category, tables, op-trace, etc) - these are always consistent and include only operations that are present in the ROI.
In interactive mode, the ROI can be changed, and this will force re-computation of the tables.

#### Explicit ROI
--roi-start, --roi-end parms explicity specify the start and end ROI, supporting a variety of different formats and syntactic sugar.
The first time in the trace is always 0.    Start timestamps must be <= end timestamps.

##### ROI Units
The default unit is 'ms', but ns, us, s, or sec may also be used. These all specify 123.45ms:
   - `--roi-start 123.45`
   - `--roi-start 123.45ms`
   - `--roi-start 123450ns`
   - `--roi-start .12345s`

##### Start- and End- Relative 
Use a leading plus sign to specify the timestamp is relative to the trace start (always 0, so this is same as no leading +)
   - `--roi-start=+123.45`

Use a leading minus sign to indicate timestmaps is negative relative to the trace end.  Note use of the = sign so the shell can parse the negative sign correctly.
This example sets ROI to a 90ms region near the end of the trace:

   - `--roi-start=-100ms --roi-end=-10ms`

##### ROI based on kernel name regex
Specify a regular expression that matches one or more kernel names.  The ROI will be set to the first occurence of the
matching kernel name.
   - `--roi-start=Cijk_`

#### Automatic ROI

##### -auto-roi-top
Set the ROI start time for the first instance of the hottest kernel, and set the ROI end time to the start time of the last instance of that same kernel.

##### -auto-roi-trim
Remove GPU idle time at the start and end of the trace. For multiple GPUs, greedily use the largest idle periods.  If specified 


### Categories

Raptor combines kernels into higher-level categories for a better view of the "forest".  For example: 

```
Categories:
           UniqKernels  TotalCalls  TotalDur_ms  VarSum_ms  AvgDur_us  Pct
GEMM                11      164673       3574.0      210.8       21.7 42.0
_Collective         10       87722       2028.6      255.3       23.1 23.8
_Other              28      292485       1745.2      443.2        6.0 20.5
Attention            2       81920        706.6       69.3        8.6  8.3
_GPU_Idle            6      136769        317.4        0.0        2.3  3.7
aten                25        9903        142.2       10.2       14.4  1.7
```

#### Builtin Categories
Raptor has several builtin category definitions which start with a leading underscore:
- _Collective : Collective communication kernels such as AllReduce or AllGather.  These include a barrier synchronization.
Raptor treats COMM kernels special for variability calculations.
- _GPU_Idle : GPU is idle
- _Variability : Attempt to categorize the gap between the fastest instance of a specific kernel vs the actual implementation.
- _Other : Kernels which don't match any defined category.  

#### Custom Categories
Categories are specified in a dictionary-like JSON file with the name of the category and corresponing list of kernel name regular expressions.
The kernelseq view includes a "Category" column.  Kernels which don't match any defined category are shown as "_Other".

The default file is show as an example below.  Users can add new categories (or expand the regex for existing ones) to get more clarity on "_Other" kernels.
This can be useful if the kernelseq list shows a large percentage of kernels in the "_Other" category.
If a kernel matches multiple categories, the LAST category in the file wins - so add more specific categories at the END of the file.
If using the variability metrics, ensure that the special "_Collective" category maps to all kernels used for barrier communication between GPUs.
```
$ cat raptor_cat_vllm.json
{
    "GEMM" : ["^Cijk_"],
    "CopyD2D" : ["CopyDeviceToDevice"],
    "aten" : ["^.* at::"],
    "_Collective" : ["ncclDevKernel_Generic","void vllm::cross_device_reduce"],
    "topk" : ["^.*topk"],
    "Attention" : ["paged_attention*", "attn_fwd"]
}
```
### Gaps and GPU-Idle
Raptor computes the time before each kernel when the GPU is idle - this can be useful to detect underutilization due to exposed 
host activity or synchronization or other causes.
Gaps are shown in the kernelseq_df summary ("PreGap_ns"), and in each record of the op-trace.
In the category summary, gaps are accumlated into the "_GPU_Idle" field.
Short gaps (<5us) are typical between kernels and indicate the expected gap between the end of the preceding kernel and the start of the next.
Gaps in the 10us-20us may point to host synchronization events which drain the GPU input queue and expose the host to device kernel launch time.
Long gaps (>1ms) can indicate exposed host activity or runtime overheads.
User can specify the "bins" for the gaps - this can be useful to more precisely bucketize the gaps.


### Variability
Execution time for the same kernel can vary due to fluctations in clock frequency, dynamic power management effects, micro-architecture differences, caches and TLB hit rates, or other factors.
Variability is computed for each kernelseq and displayed in the kernelseq_df.
Kernel sequences can be used to disambiguate kernels which have the same name - this is critical for the variability analysis to work correctly.
The fastest running instance not marked as an outlier is used as the "golden" target performance.
The "VarSum_ns" column in the kernelseq_df contains the sum of the deltas from the "golden" target for all kernel instances.

#### Aggregation with `category_df`
Aggregation with category_df in Raptor is designed to avoid double-counting when calculating variability. There are two approaches for this:

- Collective Approach: Variability is calculated using only the kernels in the "_Collective" category, which represents operations that are grouped to avoid redundancy.

- Non-Collective Approach: Variability is calculated using all kernels except those in the "_Collective" category, providing an aggregate of non-collective operations.

The variability_method parameter to get_category_df() allows users to specify which approach to use:

- None: No row is added to represent variability.  Variability time is shown as a column in the Category report.
- collective: Only the "_Collective" category is aggregated into the _Variability row.
- non_collective: All categories except "_Collective" are aggregated into the _Variability row.

Both methods assume that each GPU receives an equal workload, meaning they are expected to complete in the same amount of time.



#### Outlier detection
Raptor has a built-in mechanism to detect and remove outlier kernel instances using a z-score-based approach. The z-score, calculated as `(value - Mean) / StandardDeviation`, measures how far a data point deviates from the mean in terms of standard deviations. In a normal distribution, 99.7% of the values have an absolute z-score ≤ 3.

If an instance’s absolute z-score exceeds a user-defined threshold, Raptor removes that instance from the associated KernelSeq before calculating the KernelSeq's derived statistics (including minimum, maximum, mean, and variability metrics).

- **Op-Trace Display:** Although outlier instances are removed from KernelSeq calculations, the op-trace display retains all records and marks outliers with the `OUTLR` tag.
- **Default Inclusion:** By default, all instances are included (where `zscore == -1`), meaning no outliers are removed unless specified by the user.

This approach allows Raptor to provide cleaner, more reliable statistical insights for KernelSeqs by reducing the influence of extreme values.

## Reference
### Directory Structure
- raptor_parser.py : Class for reading and parsing RPD files.
- raptor.py : Command-line driver.
- ./cookbook : Interactive python examples and recipes showing how to use the RaptorParser API.
- ./tests : PyTest-format tests and test inputs. 

### Terminology
- Op (Operation): Each unique execution of a kernel or data movement is considered an "op." An op has a distinct sequence ID (starting at "1"), start and stop timestamps, the kernel name, and various derived statistics (e.g., Duration, PreGap).
- Kernel: A group of ops that share the same kernel name are aggregated into "kernel" records within the kernelseq_df.
- KernelSeq: A unique sequence of one or more kernels. Kernels include information such as the kernel name, a list of associated ops, and derived statistics (e.g., the time of the first and last instance, variability statistics, total PreGap, etc.).
- Instance: Each individual op within a given KernelSeq is called an "instance" of that KernelSeq.
- Category: Categories are collections of kernels with similar functionality, grouping kernels that perform related types of operations or serve similar purposes.

In essence:

- "Ops" are individual execution points.
- "KernelSeq" group ops with the same sequence of kernels. A sequence of 1 groups ops with the same kernel name.
- "Instances" are the specific occurrences of ops within a kernel.
- "Categories" group related kernels by functionality.

### Testing and Contribution Guidelines

   - Ensure all tests pass Green before submitting.
   - Add new tests for new features, following the format of the existing tests.
   - To run the tests:
```
$ pytest tests/
```
