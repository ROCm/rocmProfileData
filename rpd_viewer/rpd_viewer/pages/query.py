import time
import sqlite3

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_ag_grid as dag

from rpd_viewer.util import db

dash.register_page(__name__, path="/query", name="SQL Query")

ROW_LIMIT = 1000
QUERY_TIMEOUT_S = 10

EXAMPLE_QUERIES = [
    "SELECT apiName, start, end,\n       (end - start) / 1000.0 AS duration_us, args\nFROM api\nLIMIT 1000",
    "SELECT description, start, end,\n       (end - start) / 1000.0 AS duration_us, opType\nFROM op\nLIMIT 1000",
    "SELECT *\nFROM api\nJOIN rocpd_kernelapi ON api_ptr_id = api.id\nLIMIT 1000",
    "SELECT *\nFROM api\nJOIN rocpd_copyapi ON api_ptr_id = api.id\nLIMIT 1000",
    "SELECT * FROM rocpd_metadata",
]


def _get_readonly_connection():
    conn = sqlite3.connect(f"file:{db.rpd_path}?mode=ro", uri=True, timeout=QUERY_TIMEOUT_S)
    conn.execute("PRAGMA query_only = ON")
    return conn


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    example_labels = ["API Calls", "GPU Ops", "Kernel API", "Copy API", "Metadata"]

    example_buttons = html.Div([
        html.Span("Examples: ", style={"fontSize": "12px", "color": "#888", "marginRight": "6px"}),
    ] + [
        html.Button(
            example_labels[i],
            id={"type": "example-btn", "index": i},
            n_clicks=0,
            style={
                "fontSize": "11px",
                "padding": "2px 8px",
                "marginRight": "6px",
                "cursor": "pointer",
                "border": "1px solid #ccc",
                "borderRadius": "4px",
                "backgroundColor": "#f5f5f5",
            },
        ) for i in range(len(EXAMPLE_QUERIES))
    ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"})

    return html.Div([
        html.H2("SQL Query"),
        html.P("Read-only access to the RPD database.",
               style={"color": "#666", "marginBottom": "16px"}),
        example_buttons,
        dcc.Textarea(
            id="sql-input",
            value=EXAMPLE_QUERIES[0],
            style={
                "width": "100%",
                "height": "120px",
                "fontFamily": "monospace",
                "fontSize": "13px",
                "padding": "10px",
                "border": "1px solid #ccc",
                "borderRadius": "6px",
                "resize": "vertical",
            },
        ),
        html.Div([
            html.Button("Run", id="run-query-btn", n_clicks=0, style={
                "padding": "8px 24px",
                "fontSize": "14px",
                "backgroundColor": "#1a73e8",
                "color": "white",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer",
                "marginRight": "12px",
            }),
            html.Div([
                html.Button(
                    "Copy SQL",
                    style={
                        "fontSize": "11px", "padding": "4px 10px", "cursor": "pointer",
                        "border": "1px solid #ccc", "borderRadius": "4px",
                        "backgroundColor": "#f5f5f5", "marginRight": "8px",
                    },
                    **{
                        "data-x-on:click": (
                            "navigator.clipboard.writeText("
                            "document.getElementById('sql-input').value); "
                            "copied = true; setTimeout(() => copied = false, 1500)"
                        ),
                    },
                ),
                html.Span("Copied!", style={"color": "#1a9e5c", "fontSize": "12px"},
                          **{"data-x-show": "copied", "data-x-transition": ""}),
            ], **{"data-x-data": "{ copied: false }"}, style={"display": "flex", "alignItems": "center"}),
            html.Span(id="query-status", style={"fontSize": "13px", "color": "#666", "marginLeft": "12px"}),
        ], style={"marginTop": "10px", "marginBottom": "20px", "display": "flex", "alignItems": "center"}),
        dcc.Loading(type="circle", children=html.Div(id="query-results")),
    ])


@callback(
    Output("sql-input", "value"),
    [Input({"type": "example-btn", "index": dash.ALL}, "n_clicks")],
    prevent_initial_call=True,
)
def load_example(n_clicks_list):
    ctx = dash.ctx
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        return dash.no_update
    return EXAMPLE_QUERIES[ctx.triggered_id["index"]]


@callback(
    Output("query-results", "children"),
    Output("query-status", "children"),
    Input("run-query-btn", "n_clicks"),
    State("sql-input", "value"),
    prevent_initial_call=True,
)
def run_query(n_clicks, sql):
    if not sql or not sql.strip():
        return html.Div(), "Enter a query."

    sql = sql.strip().rstrip(";")

    t0 = time.time()
    try:
        conn = _get_readonly_connection()
        try:
            cur = conn.execute(sql)
            if cur.description is None:
                elapsed = time.time() - t0
                return html.Div(), f"Statement executed ({elapsed:.3f}s). No results returned."

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(ROW_LIMIT + 1)
            elapsed = time.time() - t0

            truncated = len(rows) > ROW_LIMIT
            if truncated:
                rows = rows[:ROW_LIMIT]

            row_dicts = [dict(zip(columns, row)) for row in rows]
        finally:
            conn.close()

        col_defs = [{"field": c, "headerName": c} for c in columns]

        status_parts = [f"{len(row_dicts):,} rows"]
        if truncated:
            status_parts.append(f"(limited to {ROW_LIMIT:,})")
        status_parts.append(f"in {elapsed:.3f}s")
        status = " ".join(status_parts)

        grid = dag.AgGrid(
            rowData=row_dicts,
            columnDefs=col_defs,
            defaultColDef={"sortable": True, "resizable": True, "filter": True},
            dashGridOptions={"rowHeight": 28, "headerHeight": 32},
            style={"height": f"{min(len(row_dicts) * 30 + 42, 600)}px"},
        )

        return grid, status

    except sqlite3.OperationalError as e:
        elapsed = time.time() - t0
        msg = str(e)
        return html.Div(
            f"Error: {msg}",
            style={
                "padding": "12px 16px",
                "backgroundColor": "#fef2f2",
                "color": "#b91c1c",
                "borderRadius": "6px",
                "fontFamily": "monospace",
                "fontSize": "13px",
            },
        ), f"Failed ({elapsed:.3f}s)"
    except Exception as e:
        elapsed = time.time() - t0
        return html.Div(
            f"Error: {e}",
            style={
                "padding": "12px 16px",
                "backgroundColor": "#fef2f2",
                "color": "#b91c1c",
                "borderRadius": "6px",
                "fontFamily": "monospace",
                "fontSize": "13px",
            },
        ), f"Failed ({elapsed:.3f}s)"
