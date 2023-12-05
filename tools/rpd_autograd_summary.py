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
# Format sqlite trace data as json for chrome:tracing
#

import sys
import os
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='Format autograd kernel usage as html')
parser.add_argument('input_rpd', type=str, help="input rpd db")
parser.add_argument('output_html', type=str, help="html output")
parser.add_argument('--start', type=int, help="start timestamp")
parser.add_argument('--end', type=int, help="end timestamp")
args = parser.parse_args()

connection = sqlite3.connect(args.input_rpd)

outfile = open(args.output_html, 'w', encoding="utf-8")

outfile.write("""
<!DOCTYPE html>
<html>
<head>
</head>
<body>


<h2>Kernels by Autograd Operator</h2>
<p></p>


<style>
th {
  text-align: left;
}

.caret {
  cursor: pointer;
  background-color: rgb(230,230,230);
  -webkit-user-select: none; /* Safari 3.1+ */
  -moz-user-select: none; /* Firefox 2+ */
  -ms-user-select: none; /* IE 10+ */
  user-select: none;
}

.caret::before {
  content: "\\25B6";
  color: black;
  display: inline-block;
  margin-right: 6px;
}

.caret-down::before {
  -ms-transform: rotate(90deg); /* IE 9 */
  -webkit-transform: rotate(90deg); /* Safari */'
  transform: rotate(90deg);  
}

.nested {
  display: none;
}

.active {
  display: table-row;
}
</style>

<table>
  <tr>
    <th>
    <th>Autograd Operator</th>
    <th>Kernel Name</th>
    <th>Sizes</th>
    <th>Total Call</th>
    <th>Average GPU</th>
    <th>Total GPU</th>
  </tr>
""")


last_labels = ["",""]
group_active = [False, False]

for row in connection.execute('select "0" as ordinal, count(*) as count, autogradName, "" as kernelName, "" as Sizes, sum(calls) as calls, sum(avg_gpu) as avg_gpu, sum(total_gpu) as total_gpu from autogradKernel group by autogradName UNION ALL select "1" as ordinal, count(*) as count, autogradName, kernelName, "" as Sizes, sum(calls) as calls, sum(avg_gpu) as avg_gpu, sum(total_gpu) as total_gpu from autogradKernel group by autogradName, kernelName UNION ALL select "2" as ordinal, count(*) as count, autogradName, kernelName, Sizes, sum(calls) as calls, sum(avg_gpu) as avg_gpu, sum(total_gpu) as total_gpu from autogradKernel group by autogradName, kernelName, sizes order by autogradName, kernelName, ordinal'):

    depth = int(row[0])
    count = int(row[1])
    labels = [row[2], row[3]]

    #outfile.write(f'         {depth} {count} {labels[0][:80]} {labels[1][:80]}')
    #doit = "now";

    # Detect start of groupings
    #for i in [0,1]:
    #    if depth == i and count > 1:
    #        outfile.write("<tbody>")
    #        group_active[i] = True

    # Detect end of groupings
    for i in [1,0]:
        changed = False
        for j in range(i + 1):
            if labels[i] != last_labels[i]:
                changed = True
        if changed and group_active[i]:
            outfile.write("</tbody>")
            group_active[i] = False

    # Detect start of groupings
    for i in [0,1]:
        if depth == i and count > 1:
            outfile.write("<tbody>")
            group_active[i] = True

    # Don't outfile.write out single leaf branches
    if depth < 2 and count < 2:
        continue

    # supress labels for duplicates
    for i in [0,1]:
        if group_active[i] and labels[i] == last_labels[i]:
            labels[i] = ""
        else:
            last_labels[i] = labels[i]

    class_string = "><td></td"
    if group_active[0] or group_active[1]: class_string = ' class="nested"><td></td'
    for i in [0,1]:
        if depth == i and group_active[i]: class_string = ' class="caret"'
    outfile.write(f'<tr{class_string}><td>{labels[0]}</td><td>{labels[1][:60]}</td><td>{row[4]}</td><td>{row[5]}</td><td>{row[6]}</td><td>{row[7]}</td></tr>')

#clean up groupings
for i in [1,0]:
    if group_active[i]:
            outfile.write("</tbody>")
            group_active[i] = False

outfile.write("""
</table>
<script>
var toggler = document.getElementsByClassName("caret");
var i;

for (i = 0; i < toggler.length; i++) {
  toggler[i].addEventListener("click", function() {
    rows = this.parentElement.querySelectorAll(".nested")
    for (i = 0; i < rows.length; i++)
        rows[i].classList.toggle("active");
    this.classList.toggle("caret-down");
  });
}
</script>
</body>
</html>
""")

outfile.write("\n")
outfile.close()
connection.close()
