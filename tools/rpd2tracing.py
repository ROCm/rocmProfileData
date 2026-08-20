#!/usr/bin/env python3

# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#
# Format sqlite trace data as json for chrome:tracing
#

import sqlite3
import argparse
import pathlib
from rocpd.tracing import generate_rpd_json

parser = argparse.ArgumentParser(description='convert RPD to json for chrome tracing')
parser.add_argument('input_rpd', type=str, help="input rpd db")
parser.add_argument('output_json', type=str, nargs='?', help="chrome tracing json output")
parser.add_argument('--start', type=str, help="start time - default us or percentage %%. Number only is interpreted as us. Number with %% is interpreted as percentage")
parser.add_argument('--end', type=str, help="end time - default us or percentage %%. See help for --start")
parser.add_argument('--format', type=str, default="object", help="chrome trace format, array or object")
args = parser.parse_args()

if args.output_json is None:
    args.output_json = str(pathlib.PurePath(args.input_rpd).with_suffix(".json"))

connection = sqlite3.connect(args.input_rpd)

with open(args.output_json, 'w', encoding="utf-8") as outfile:
    generate_rpd_json(connection, outfile,
                      start=args.start, end=args.end,
                      trace_format=args.format,
                      verbose=True)

connection.close()
