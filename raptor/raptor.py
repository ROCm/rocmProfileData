#!/usr/bin/env python3

import argparse
import pathlib
import os
import pandas as pd
import numpy as np

from raptor_parser import RaptorParser

parser = argparse.ArgumentParser(prog="raptor.py",
        description=
            RaptorParser.usage_doc + \
"""
Example:  $ raptor.py --top --categorize trace.rpd
""",
            formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("rpd_file_name")

summary_g = parser.add_argument_group('Categorize and Top Kernel Summaries')
summary_g.add_argument("--categorize", "-c", action='store_true',
                    help="Summarize RPD top kernels into categories (ie GEMM, AllReduce, EltWise, Idle)")
summary_g.add_argument("--category-json", "-C", type=str,
                    default=os.path.join(pathlib.Path(__file__).parent.resolve(), "raptor_cat_vllm.json"),
                    help="File containing category definitions, specified as a JSON-format dictionary.  See tools/raptor_cat_vllm.json for an example.  If a kernel name matches more than one pattern, the LAST match in the file determines the category.")
summary_g.add_argument("--variability", "-v", action='store_true',
                    help="Show variability df")
summary_g.add_argument("--top", "-t", action='store_true',
                    help="Show top kernels, sorted by TotalDuration")
summary_g.add_argument("--prekernel-seq", type=int, default=2,
                    help="Number of preceding kernels to use in the sequence of kernels used for grouping into the top buckets.  This can be used to disambiguate cases where the same kernel name is called in different contents.  0 means to ignore the sequence and aggregate top kernels just based on the name")
summary_g.add_argument("--gaps",
                nargs='+', type=int, metavar="GAP",
                default=RaptorParser.default_gaps,
                help = "Size of histogram buckets used for gaps breakdown, specified as a list of micro-second times.  Default="+str(RaptorParser.default_gaps));

summary_g.add_argument("--zscore_threshold", "-z", type=int,
                        default=RaptorParser.default_zscore,
                        help="Zscore threshold to use to identify outliers for each kernel in the top_df.  Raptor computes the zscore=(val - Mean)/StdDev for the Duration of each instance of each kernel.  if abs(zscore)>zscore_threshold, the instance is treated as an outlier and excluded from the top_df stats.  For a normal distribution, zscore_threshold==3 captures 99.7%% of the values and can be a good starting value. Default zscore is -1 (outlier detection is disabled)")

op_trace_g = parser.add_argument_group('Op-Trace')
op_trace_g.add_argument("--op-trace", "-o", action='store_true',
                    help="Generate a single-line trace for each op(kernel) showing pre-gap, start/end times, duration, name, etc.")
op_trace_g.add_argument("--op-trace-file", type=str,
                    help="Write op-trace to a file.  If not specified, write to stdout.")
op_trace_g.add_argument("--op-trace-cmd-width", '-W', 
                    type=int, default=None,
                    help="Width in characters to display the op (kernel) names in the op trace")
op_trace_g.add_argument("--instance-trace", "-i", type=int, default=None,
                    help="Show each execution of the specified kernel.  The parm is an integer index into the 'top_df' table - ie 0 is the first row, 1 is the second, and so on.  Negative indices start from the bottom of the top_df table.  Instances are sorted by Duration_ns")

roi_g = parser.add_argument_group('Region-of-Interest (ROI)')
roi_g.add_argument("--roi-start", "-s", type=str,
                    help="Set ROI start. 0 corresponds to the start of the RPD.  Default is ms, but can specify trailing time units.  If kernel name is specified, use the timestamp of the first instance for the specified kernel. Examples: 123.45, 123.45ms, 123450ns, .12345s, Cijk_")
roi_g.add_argument("--roi-end", "-e", type=str,
                    help="Set Region-of-Interest end. See --start for format.")
roi_g.add_argument("--auto-roi-top", action='store_true',
                    help="Automatically pick the ROI to include first and last instance of the hottest duration kernel")
roi_g.add_argument("--gpu-id", type=int, default=RaptorParser.default_gpu_id,
                    help="Only show records from the specified gpuId.  (Default or -1: combine ops from all GPUs)")

display_g = parser.add_argument_group('Display arguments')
display_g.add_argument("--display-cols", type=int, default=60,
                    help="Set display column width")
display_g.add_argument("--display-rows", type=int, default=500,
                    help="Set number of rows")
display_g.add_argument("--float-digits", type=int, default=1,
                    help="Number of digits to print for float values in tables.")

group = parser.add_argument_group('Output controls')
group.add_argument("--write-xls", '-x', action='store_true',
                    help="Write XLS file with key tables")
group.add_argument("--write-xls-op-trace", action='store_true',
                    help="Write the full op trace to the XLS")

args=parser.parse_args()

if args.write_xls_op_trace:
    args.write_xls = True

# Set display options:
pd.set_option('display.max_rows', args.display_rows)
pd.options.display.max_colwidth = args.display_cols
pd.set_option('display.float_format', ('{:.%df}'%args.float_digits).format)

if not args.op_trace and not args.top and not args.categorize \
   and args.instance_trace is None and not args.write_xls:
    print ("info: setting --top --categorize --variability")
    args.top = True
    args.categorize = True
    args.variability = True

raptor = RaptorParser(args.rpd_file_name, gaps=args.gaps, 
                      category_json=args.category_json,
                      prekernel_seq=args.prekernel_seq,
                      gpu_id=args.gpu_id,
                      zscore_threshold=args.zscore_threshold,
                      roi_start=args.roi_start, roi_end=args.roi_end)

raptor.print_timestamps(indent="   ")

if args.auto_roi_top:
    raptor.set_roi_from_kernel()

if args.categorize: 
    category_df = raptor.get_category_df(raptor.get_top_df())
    print ("\nCategories:")
    print(category_df)

if args.variability: 
    print ("\nVariability:")
    print(raptor.get_variability_df())

if args.top:
    print ("\nTop Kernels:")
    print (raptor.get_pretty_top_df())

if args.op_trace:
    raptor.print_op_trace(outfile=args.op_trace_file, command_print_width=args.op_trace_cmd_width)

if args.instance_trace is not None:
    kernel_df = raptor.get_top_df().iloc[args.instance_trace]
    instance_df = raptor.get_instance_df_from_kernel_df(kernel_df)
    raptor.print_op_trace(op_df=instance_df, outfile=args.op_trace_file, command_print_width=args.op_trace_cmd_width)

if args.write_xls:
    raptor.to_xls(add_op_df=args.write_xls_op_trace)
