################################################################################
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

#
# Create a csv summary table from an rpd file
#

import sys
import os
import re
import sqlite3
from datetime import datetime
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='convert an RPD file to a summary table in CSV')
parser.add_argument('input_rpd', type=str, help="Input RPD file")
parser.add_argument('user_marker', type=str, help="Input User Maker; leave it empty to process the entire file.")
parser.add_argument('output_csv', type=str, help="Output Summary Table to CSV")
args = parser.parse_args()

def process_rpd_to_df(rpd_path, markers_list):
    connection = sqlite3.connect(rpd_path)

    # Keep it here for now for future extension
    rangeStringApi = ""
    rangeStringOp = ""

    create_marker_list = f"""CREATE TEMPORARY TABLE ext_marker ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "pid" integer NOT NULL, "tid" integer NOT NULL, "name" varchar(255) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL)"""
    connection.execute(create_marker_list)

    markers_list = markers_list[0].split(", ")
    marker_list_update_query = f"""
        INSERT INTO ext_marker (pid, tid, name, start, end)
        SELECT pid, tid, args, start, end
        FROM api
        WHERE apiName = 'UserMarker' AND args IN ({', '.join(['?'] * len(markers_list))})
    """
    connection.execute(marker_list_update_query, markers_list)

    collect_api_query = f"""
        CREATE TEMPORARY VIEW marker as 
        SELECT A.pid, A.tid, A.id as marker_id, A.name as marker_name, B.id as api_id, apiName 
        FROM ext_marker A join api B 
        ON B.start >= A.start and B.end <= A.end and A.pid = B.pid and A.tid and B.tid
    """
    connection.execute(collect_api_query)

    collect_kernel_query = f"""
        CREATE TEMPORARY VIEW marker_kernel as 
        SELECT B.*, gpuid, C.optype as optype, C.description as kernel_name, (end - start) as duration 
        FROM rocpd_api_ops A join marker B on B.api_id = A.api_id join op C on C.id = A.op_id
    """
    connection.execute(collect_kernel_query)

    generate_table_query = f"""
        SELECT marker_name, marker_id, gpuid, kernel_name, COUNT(DISTINCT marker_id) as marker_count, count(duration) as kernel_count, sum(duration) as total_dur, avg(duration) as avg_dur, min(duration) as min_dur, max(duration) as max_dur,
        (SUM(duration) * 100.0 / SUM(SUM(duration)) OVER (PARTITION BY marker_name, gpuid)) AS kernel_percentage
        FROM marker_kernel 
        GROUP BY marker_name, gpuid, kernel_name
        ORDER BY gpuid
    """

    table_df = pd.read_sql_query(generate_table_query, connection)

    return table_df


def main():
    parser = argparse.ArgumentParser(description='Convert an RPD file to a summary table in CSV')
    parser.add_argument('input_rpd', type=str, help="Input RPD file")
    parser.add_argument('user_marker', type=str, nargs='*', help="Input User Marker(s); leave it empty to process the entire file.")
    parser.add_argument('output_csv', type=str, help="Output Summary Table to CSV")
    args = parser.parse_args()

    rpd_path = args.input_rpd
    markers_list = args.user_marker
    output_csv = args.output_csv

    table_df = process_rpd_to_df(rpd_path, markers_list)

    table_df.to_csv(output_csv, index=False)
    print(f"Summary table saved to {output_csv}")

if __name__ == "__main__":
    main()