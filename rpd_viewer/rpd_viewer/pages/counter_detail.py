import dash
from dash import html, dcc, callback, Input, Output
import dash_ag_grid as dag
import pandas as pd

from rpd_viewer.util import db

dash.register_page(__name__, path="/counters/detail", name="Counter Detail")

DETAIL_SQL = """
SELECT A.op_id, (D."end" - D.start) / 1000.0 AS duration_us,
       B.string AS counterName, A.value
FROM rocpd_counter A
JOIN rocpd_string B ON B.id = A.name_id
JOIN rocpd_op D ON D.id = A.op_id
JOIN rocpd_string C ON C.id = D.description_id
WHERE C.string = ?
ORDER BY A.op_id, counterName
"""


def layout(kernel=None, **_kwargs):
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not kernel:
        return html.Div([
            html.H2("Counter Detail"),
            html.P("No kernel specified."),
            dcc.Link("Back to Counters", href="/counters"),
        ])

    try:
        df = db.query_df(DETAIL_SQL, params=(kernel,))
        if df.empty:
            return html.Div([
                html.H2("Counter Detail"),
                html.P(f"No counter data found for this kernel."),
                dcc.Link("Back to Counters", href="/counters"),
            ])

        pivot = df.pivot_table(
            index=["op_id", "duration_us"],
            columns="counterName",
            values="value",
            aggfunc="first",
        ).reset_index()
        pivot.columns.name = None

        counter_cols = [c for c in pivot.columns if c not in ("op_id", "duration_us")]

        for c in counter_cols:
            pivot[c] = pivot[c].round(2)
        pivot["duration_us"] = pivot["duration_us"].round(2)

        col_defs = [
            {"field": "op_id", "headerName": "Op ID", "pinned": "left", "width": 100},
            {"field": "duration_us", "headerName": "Duration (us)", "width": 140,
             "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        ]
        for c in sorted(counter_cols):
            col_defs.append({
                "field": c,
                "headerName": c,
                "width": 140,
                "valueFormatter": {
                    "function": "params.value == null ? '—' : d3.format(',.2f')(params.value)"
                },
            })

        rows = pivot.where(pivot.notna(), None).to_dict("records")

        truncated = kernel[:100] + "..." if len(kernel) > 100 else kernel

        return html.Div([
            dcc.Link("← Back to Counters", href="/counters",
                     style={"color": "#1a73e8", "textDecoration": "none", "fontSize": "14px"}),
            html.H2("Counter Detail", style={"marginTop": "10px"}),
            html.Div(truncated, title=kernel, style={
                "fontFamily": "monospace",
                "fontSize": "13px",
                "color": "#555",
                "marginBottom": "6px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
                "maxWidth": "100%",
            }),
            html.P(f"{len(rows)} dispatches, {len(counter_cols)} counters",
                   style={"color": "#888", "fontSize": "13px", "marginBottom": "16px"}),
            dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=rows,
                columnDefs=col_defs,
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": f"{min(len(rows) * 30 + 42, 700)}px"},
            )),
        ])
    except Exception as e:
        return html.Div(f"Error loading counter detail: {e}")
