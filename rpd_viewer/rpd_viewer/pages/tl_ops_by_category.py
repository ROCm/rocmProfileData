import re
import dash
from dash import html, dcc
import dash_ag_grid as dag
import plotly.express as px
import pandas as pd

from rpd_viewer.util import db

dash.register_page(__name__, path="/tl/ops-by-category", name="Ops by Category")

OP_CATEGORY_MAP = [
    ("GEMM", [r"addmm", r"matmul", r"mm$", r"bmm", r"linear"]),
    ("Convolution", [r"conv2d", r"conv1d", r"conv3d", r"convolution", r"miopen_convolution"]),
    ("BatchNorm", [r"batch_norm", r"miopen_batch_norm"]),
    ("Normalization", [r"layer_norm", r"group_norm", r"rms_norm"]),
    ("Attention", [r"attention", r"flash_attn", r"sdpa", r"scaled_dot"]),
    ("Activation", [r"relu", r"silu", r"gelu", r"sigmoid", r"tanh", r"clamp_min", r"threshold"]),
    ("Pooling", [r"pool", r"adaptive_avg"]),
    ("Embedding", [r"embedding"]),
    ("Dropout", [r"dropout"]),
    ("Loss", [r"loss", r"nll_loss", r"cross_entropy", r"softmax"]),
    ("Optimizer", [r"Optimizer\.", r"foreach_add", r"foreach_mul"]),
    ("DataMovement", [r"copy_", r"to_copy", r"contiguous", r"clone", r"cat$"]),
    ("View/Reshape", [r"view", r"reshape", r"permute", r"transpose", r"expand", r"as_strided", r"unsqueeze", r"squeeze"]),
]


def _categorize_op(name):
    for cat, patterns in OP_CATEGORY_MAP:
        for p in patterns:
            if re.search(p, name, re.IGNORECASE):
                return cat
    return "Other"


TORCH_OPS_SQL = """
SELECT apiName, category as fwd_bwd, count(*) as calls,
       sum(end - start) / 1000 as cpu_time_us
FROM api
WHERE domain = 'torch' AND category IN ('function', 'backward_function')
GROUP BY apiName, fwd_bwd
ORDER BY cpu_time_us DESC
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not db.has_torch_ops():
        return html.Div([html.H2("Ops by Category"), html.P("No PyTorch annotations in this trace.")])

    try:
        df = db.query_df(TORCH_OPS_SQL)
        df["op_category"] = df["apiName"].apply(_categorize_op)

        cat_df = df.groupby("op_category").agg(
            unique_ops=("apiName", "nunique"),
            total_calls=("calls", "sum"),
            total_cpu_us=("cpu_time_us", "sum"),
        ).reset_index()
        total = cat_df["total_cpu_us"].sum()
        cat_df["pct"] = cat_df["total_cpu_us"] * 100.0 / total if total > 0 else 0
        cat_df = cat_df.sort_values("total_cpu_us", ascending=False)
        cat_df["cum_pct"] = cat_df["pct"].cumsum()

        fig = px.pie(cat_df, values="total_cpu_us", names="op_category",
                     title="CPU Time by Op Category", hole=0.4)
        fig.update_layout(height=400)

        fmt_num = {"function": "d3.format(',')(params.value)"}

        return html.Div([
            html.H2("Ops by Category"),
            dcc.Loading(type="circle", children=html.Div([
                dcc.Graph(figure=fig),
                html.H3("Category Summary"),
                dag.AgGrid(
                    rowData=cat_df.to_dict("records"),
                    columnDefs=[
                        {"field": "op_category", "headerName": "Category", "flex": 2},
                        {"field": "unique_ops", "headerName": "Unique Ops", "flex": 1, "valueFormatter": fmt_num},
                        {"field": "total_calls", "headerName": "Total Calls", "flex": 1, "valueFormatter": fmt_num},
                        {"field": "total_cpu_us", "headerName": "CPU Time (us)", "flex": 1, "valueFormatter": fmt_num},
                        {"field": "pct", "headerName": "%", "flex": 1,
                         "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
                        {"field": "cum_pct", "headerName": "Cum %", "flex": 1,
                         "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
                    ],
                    defaultColDef={"sortable": True, "resizable": True},
                    dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                    style={"height": "450px"},
                ),
                html.H3("All Ops", style={"marginTop": "25px"}),
                dag.AgGrid(
                    rowData=df.to_dict("records"),
                    columnDefs=[
                        {"field": "op_category", "headerName": "Category", "flex": 1, "rowGroup": True, "hide": True},
                        {"field": "apiName", "headerName": "Op", "flex": 3},
                        {"field": "fwd_bwd", "headerName": "Fwd/Bwd", "flex": 1},
                        {"field": "calls", "headerName": "Calls", "flex": 1, "valueFormatter": fmt_num},
                        {"field": "cpu_time_us", "headerName": "CPU Time (us)", "flex": 1, "valueFormatter": fmt_num},
                    ],
                    defaultColDef={"sortable": True, "resizable": True, "filter": True},
                    dashGridOptions={"rowHeight": 28, "headerHeight": 32, 
                        "groupDefaultExpanded": 0,
                        "autoGroupColumnDef": {"headerName": "Category", "minWidth": 250},
                    },
                    style={"height": "600px"},
                ),
            ])),
        ])
    except Exception as e:
        return html.Div(f"Error loading ops by category: {e}")
