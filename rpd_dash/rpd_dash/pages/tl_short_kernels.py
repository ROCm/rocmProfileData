import dash
from dash import html, dcc, callback, Input, Output
import dash_ag_grid as dag
import plotly.express as px
import pandas as pd

from rpd_dash.util import db

dash.register_page(__name__, path="/tl/short-kernels", name="Short Kernels")

_BARRIER_FILTER = "opType_id NOT IN (SELECT id FROM rocpd_string WHERE string = 'Barrier')"

SHORT_KERNEL_SQL = f"""
SELECT C.string as Name, count(*) as Calls,
       sum(A.end - A.start) / 1000 as TotalDuration_us,
       avg(A.end - A.start) / 1000 as AvgDuration_us,
       min(A.end - A.start) / 1000 as MinDuration_us,
       max(A.end - A.start) / 1000 as MaxDuration_us
FROM (
    SELECT opType_id as name_id, start, end FROM rocpd_op
        WHERE description_id IN (SELECT id FROM rocpd_string WHERE string = '') AND {_BARRIER_FILTER}
    UNION
    SELECT description_id, start, end FROM rocpd_op
        WHERE description_id NOT IN (SELECT id FROM rocpd_string WHERE string = '') AND {_BARRIER_FILTER}
) A
JOIN rocpd_string C ON C.id = A.name_id
WHERE (A.end - A.start) / 1000 < ?
GROUP BY Name ORDER BY TotalDuration_us DESC
"""

HISTOGRAM_SQL = f"""
SELECT (rocpd_op.end - rocpd_op.start) / 1000 as duration_us
FROM rocpd_op
WHERE (rocpd_op.end - rocpd_op.start) / 1000 < ? AND {_BARRIER_FILTER}
"""

TOTAL_GPU_SQL = f"SELECT sum(end - start) / 1000 as total_us FROM rocpd_op WHERE {_BARRIER_FILTER}"


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    return html.Div([
        html.H2("Short Kernel Analysis"),
        html.Div([
            html.Label("Threshold (us): "),
            dcc.Input(id="short-threshold", type="number", value=10, min=1, max=1000,
                      style={"width": "80px", "marginLeft": "5px"}),
        ], style={"marginBottom": "20px"}),
        dcc.Loading(type="circle", children=html.Div(id="short-kernel-content")),
    ])


@callback(
    Output("short-kernel-content", "children"),
    Input("short-threshold", "value"),
)
def update_content(threshold):
    if not threshold or not db.rpd_path:
        return html.P("Enter a threshold.")

    try:
        df = db.query_df(SHORT_KERNEL_SQL, params=(threshold,))
        total_gpu = db.query_df(TOTAL_GPU_SQL)["total_us"].iloc[0]
        hist_df = db.query_df(HISTOGRAM_SQL, params=(threshold,))

        if df.empty:
            return html.P(f"No kernels shorter than {threshold} us.")

        total_short_calls = df["Calls"].sum()
        total_short_time = df["TotalDuration_us"].sum()
        total_all_calls = db.query_df(f"SELECT count(*) as cnt FROM rocpd_op WHERE {_BARRIER_FILTER}")["cnt"].iloc[0]

        summary = html.Div([
            html.P(f"Kernels < {threshold} us: {total_short_calls:,} calls "
                   f"({total_short_calls * 100.0 / total_all_calls:.1f}% of all calls), "
                   f"{total_short_time:,.0f} us total "
                   f"({total_short_time * 100.0 / total_gpu:.1f}% of GPU time)"),
        ], style={"marginBottom": "15px"})

        fig = px.histogram(hist_df, x="duration_us", nbins=min(50, threshold),
                           title=f"Duration Distribution (< {threshold} us)",
                           labels={"duration_us": "Duration (us)", "count": "Count"})
        fig.update_layout(height=300)

        fmt_num = {"function": "d3.format(',')(params.value)"}

        return html.Div([
            summary,
            dcc.Graph(figure=fig),
            dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[
                    {"field": "Name", "headerName": "Kernel", "flex": 3, "tooltipField": "Name"},
                    {"field": "Calls", "headerName": "Calls", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "TotalDuration_us", "headerName": "Total (us)", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "AvgDuration_us", "headerName": "Avg (us)", "flex": 1,
                     "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
                    {"field": "MinDuration_us", "headerName": "Min (us)", "flex": 1,
                     "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
                    {"field": "MaxDuration_us", "headerName": "Max (us)", "flex": 1,
                     "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "500px"},
            ),
        ])
    except Exception as e:
        return html.Div(f"Error: {e}")
