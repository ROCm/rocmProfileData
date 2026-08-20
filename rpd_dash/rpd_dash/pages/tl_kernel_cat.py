import re
import dash
from dash import html, dcc
import dash_ag_grid as dag
import pandas as pd

from rpd_dash.util import db

dash.register_page(__name__, path="/tl/kernel-categories", name="Kernel Categories")

CATEGORY_RULES = [
    ("GEMM", 20, [r"gemm", r"cublas", r"rocblas", r"Cijk_", r"splitK", r"matmul", r"addmm", r"bmm"]),
    ("Convolution", 20, [r"conv", r"miopenSp3AsmConv", r"miopenGcnAsmConv", r"Im2Col", r"winograd"]),
    ("BatchNorm", 20, [r"batchnorm", r"MIOpenBatchNorm", r"batch_norm"]),
    ("Normalization", 15, [r"rmsnorm", r"layernorm", r"l2norm", r"group_norm"]),
    ("Attention", 15, [r"attention", r"flash_attn", r"sdpa", r"fmha", r"softmax"]),
    ("Activation", 10, [r"relu", r"silu", r"gelu", r"swish", r"sigmoid", r"tanh", r"clamp_min"]),
    ("Elementwise", 10, [r"vectorized_elementwise", r"elementwise", r"_to_copy", r"copy_kernel", r"CatArrayBatched", r"fill_kernel"]),
    ("Reduction", 10, [r"reduce", r"sum_kernel", r"mean_kernel", r"argmax", r"topk"]),
    ("Pooling", 10, [r"pool", r"adaptive_avg", r"max_pool"]),
    ("Communication", 20, [r"nccl", r"rccl", r"allreduce", r"allgather", r"reducescatter", r"all_to_all"]),
    ("MemCopy", 20, [r"CopyHostToDevice", r"CopyDeviceToHost", r"CopyDeviceToDevice", r"FillBuffer", r"memcpy", r"memset"]),
]


def _categorize_kernel(name):
    for cat, confidence, patterns in CATEGORY_RULES:
        for p in patterns:
            if re.search(p, name, re.IGNORECASE):
                return cat
    return "Other"


TOP_SQL = """
SELECT C.string as Name, count(C.string) as TotalCalls,
    sum(A.end - A.start) / 1000 as TotalDuration_us,
    (sum(A.end - A.start) / count(C.string)) / 1000 as Ave_us,
    sum(A.end - A.start) * 100.0 / (SELECT sum(A.end - A.start) FROM rocpd_op A) as Percentage
FROM (
    SELECT opType_id as name_id, start, end FROM rocpd_op
        WHERE description_id IN (SELECT id FROM rocpd_string WHERE string = '')
    UNION
    SELECT description_id, start, end FROM rocpd_op
        WHERE description_id NOT IN (SELECT id FROM rocpd_string WHERE string = '')
) A
JOIN rocpd_string C ON C.id = A.name_id
GROUP BY Name ORDER BY TotalDuration_us DESC
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        if db.table_exists("top"):
            df = db.query_df("SELECT * FROM top")
        else:
            df = db.query_df(TOP_SQL)

        name_col = "Name"
        dur_col = next(c for c in df.columns if c.lower().startswith("totalduration"))
        calls_col = "TotalCalls"
        pct_col = "Percentage"

        df = df[df[name_col] != "Barrier"]
        total_dur = df[dur_col].sum()
        if total_dur > 0:
            df[pct_col] = df[dur_col] * 100.0 / total_dur

        df["Category"] = df[name_col].apply(_categorize_kernel)

        cat_df = df.groupby("Category").agg(
            Kernels=("Category", "count"),
            TotalCalls=(calls_col, "sum"),
            TotalDuration_us=(dur_col, "sum"),
        ).reset_index()
        total_dur = cat_df["TotalDuration_us"].sum()
        cat_df["Pct"] = cat_df["TotalDuration_us"] * 100.0 / total_dur if total_dur > 0 else 0
        cat_df = cat_df.sort_values("TotalDuration_us", ascending=False)

        fmt_num = {"function": "d3.format(',')(params.value)"}

        return html.Div([
            html.H2("Kernel Categories"),
            html.H3("By Category"),
            dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=cat_df.to_dict("records"),
                columnDefs=[
                    {"field": "Category", "headerName": "Category", "flex": 2},
                    {"field": "Kernels", "headerName": "Unique Kernels", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "TotalCalls", "headerName": "Total Calls", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "TotalDuration_us", "headerName": "Total (us)", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "Pct", "headerName": "%", "flex": 1,
                     "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
                ],
                defaultColDef={"sortable": True, "resizable": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "400px"},
            )),
            html.H3("All Kernels", style={"marginTop": "25px"}),
            dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[
                    {"field": "Category", "headerName": "Category", "flex": 1, "rowGroup": True, "hide": True},
                    {"field": name_col, "headerName": "Kernel", "flex": 3, "tooltipField": name_col},
                    {"field": calls_col, "headerName": "Calls", "flex": 1, "valueFormatter": fmt_num},
                    {"field": dur_col, "headerName": "Total (us)", "flex": 1, "valueFormatter": fmt_num},
                    {"field": pct_col, "headerName": "%", "flex": 1,
                     "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32, 
                    "groupDefaultExpanded": 0,
                    "autoGroupColumnDef": {"headerName": "Category", "minWidth": 250},
                },
                style={"height": "600px"},
            )),
        ])
    except Exception as e:
        return html.Div(f"Error loading kernel categories: {e}")
