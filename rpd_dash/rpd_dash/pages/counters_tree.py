import json

import dash
from dash import html, dcc, callback, Input, Output, State, ALL, ctx
import dash_ag_grid as dag

from rpd_dash.util import db

dash.register_page(__name__, path="/counters", name="Counters")

RANKING_SQL = """
SELECT C.string as kernelName, count(*) as TotalCalls,
    sum(A."end" - A.start) / 1000 as TotalDuration_us,
    (sum(A."end" - A.start) / count(*)) / 1000 as Avg_us,
    sum(A."end" - A.start) * 100.0 / (SELECT sum("end" - start) FROM rocpd_op) as Percentage
FROM rocpd_op A
JOIN rocpd_string C ON C.id = A.description_id
WHERE C.string IN (SELECT DISTINCT kernelName FROM counter_summary)
GROUP BY C.string
ORDER BY TotalDuration_us DESC
"""

COUNTER_SQL = """
SELECT counterName, dispatches, avg, min, max
FROM counter_summary
WHERE kernelName = ?
ORDER BY counterName
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not db.table_exists("rocpd_counter"):
        return html.Div([
            html.H2("GPU Counters"),
            html.P("No counter data in this trace."),
        ])

    try:
        df = db.query_df(RANKING_SQL)
        if df.empty:
            return html.Div([
                html.H2("GPU Counters"),
                html.P("No counter data in this trace."),
            ])

        rows = []
        for i, row in df.iterrows():
            kernel = row["kernelName"]
            truncated = kernel[:80] + "..." if len(kernel) > 80 else kernel
            idx = int(i)

            header = html.Div([
                html.Span(
                    "▶",
                    id={"type": "tree-chevron", "index": idx},
                    style={
                        "flex": "0 0 20px",
                        "cursor": "pointer",
                        "fontSize": "12px",
                        "color": "#666",
                        "transition": "transform 0.15s",
                        "userSelect": "none",
                    },
                ),
                html.Span(kernel, title=kernel, style={
                    "flex": "1 1 0",
                    "minWidth": "0",
                    "fontFamily": "monospace",
                    "fontSize": "13px",
                    "fontWeight": "500",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                }),
                _stat("Calls", f"{int(row['TotalCalls']):,}", "80px"),
                _stat("Total", f"{int(row['TotalDuration_us']):,} us", "120px"),
                _stat("Avg", f"{row['Avg_us']:,.1f} us", "100px"),
                _stat("%", f"{row['Percentage']:.2f}", "60px"),
            ], id={"type": "tree-header", "index": idx},
               n_clicks=0,
               style={
                   "padding": "10px 14px",
                   "cursor": "pointer",
                   "borderBottom": "1px solid #eee",
                   "display": "flex",
                   "alignItems": "center",
                   "overflow": "hidden",
               })

            detail_panel = html.Div(
                id={"type": "tree-panel", "index": idx},
                style={"display": "none"},
            )

            # Store kernel name so the callback can query it
            store = dcc.Store(
                id={"type": "tree-kernel", "index": idx},
                data=kernel,
            )

            rows.append(html.Div([header, detail_panel, store],
                                  style={"overflow": "hidden"}))

        return html.Div([
            html.H2("GPU Counters"),
            html.P(f"{len(df)} kernels with counter data, sorted by total GPU time",
                   style={"color": "#666", "marginBottom": "20px"}),
            html.Div(rows, style={
                "border": "1px solid #e0e0e0",
                "borderRadius": "8px",
                "backgroundColor": "#fff",
                "overflow": "hidden",
            }),
        ])
    except Exception as e:
        return html.Div(f"Error loading counters: {e}")


def _stat(label, value, width):
    return html.Span([
        html.Span(f"{label}: ", style={"color": "#999", "fontSize": "11px"}),
        html.Span(value, style={"fontSize": "12px"}),
    ], style={"flex": f"0 0 {width}", "textAlign": "right"})


@callback(
    Output({"type": "tree-panel", "index": ALL}, "style"),
    Output({"type": "tree-panel", "index": ALL}, "children"),
    Output({"type": "tree-chevron", "index": ALL}, "style"),
    Input({"type": "tree-header", "index": ALL}, "n_clicks"),
    State({"type": "tree-panel", "index": ALL}, "style"),
    State({"type": "tree-panel", "index": ALL}, "children"),
    State({"type": "tree-chevron", "index": ALL}, "style"),
    State({"type": "tree-kernel", "index": ALL}, "data"),
    prevent_initial_call=True,
)
def toggle_node(all_clicks, all_styles, all_children, all_chevron_styles, all_kernels):
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        return dash.no_update, dash.no_update, dash.no_update

    clicked_idx = ctx.triggered_id["index"]

    new_styles = list(all_styles)
    new_children = list(all_children)
    new_chevrons = list(all_chevron_styles)

    for i in range(len(all_styles)):
        if i != clicked_idx:
            continue

        currently_open = all_styles[i].get("display") != "none"

        if currently_open:
            new_styles[i] = {"display": "none"}
            new_chevrons[i] = {**all_chevron_styles[i], "transform": "rotate(0deg)"}
        else:
            new_styles[i] = {
                "display": "block",
                "padding": "12px 14px 16px 36px",
                "backgroundColor": "#fafafa",
                "borderBottom": "1px solid #eee",
            }
            new_chevrons[i] = {**all_chevron_styles[i], "transform": "rotate(90deg)"}

            if not all_children[i]:
                kernel = all_kernels[i]
                new_children[i] = _build_panel(kernel)

    return new_styles, new_children, new_chevrons


def _build_panel(kernel):
    try:
        df = db.query_df(COUNTER_SQL, params=(kernel,))
        if df.empty:
            return html.P("No counter data.")

        rows = df.to_dict("records")
        for r in rows:
            r["avg"] = round(r["avg"], 2)
            r["min"] = round(r["min"], 2)
            r["max"] = round(r["max"], 2)

        grid = dag.AgGrid(
            rowData=rows,
            columnDefs=[
                {"field": "counterName", "headerName": "Counter", "flex": 2},
                {"field": "dispatches", "headerName": "Dispatches", "flex": 1,
                 "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                {"field": "avg", "headerName": "Avg", "flex": 1,
                 "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                {"field": "min", "headerName": "Min", "flex": 1,
                 "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                {"field": "max", "headerName": "Max", "flex": 1,
                 "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
            ],
            defaultColDef={"sortable": True, "resizable": True},
            style={"height": f"{len(rows) * 30 + 42}px"},
            dashGridOptions={"rowHeight": 28, "headerHeight": 32, "domLayout": "autoHeight"},
        )

        link = dcc.Link(
            "View all dispatches →",
            href=f"/counters/detail?kernel={kernel}",
            style={
                "color": "#1a73e8",
                "textDecoration": "none",
                "fontSize": "13px",
                "display": "inline-block",
                "marginTop": "10px",
            },
        )

        return html.Div([grid, link])
    except Exception as e:
        return html.P(f"Error loading counters: {e}")
