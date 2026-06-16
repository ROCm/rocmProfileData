import dash
from dash import html, callback, Input, Output, State, dcc
import dash_ag_grid as dag

from rpd_dash.util import db

dash.register_page(__name__, path="/graphs", name="Graphs")

GRAPH_SUMMARY_SQL = """
SELECT id, graph, graphExec, stream, start, end, (end - start) / 1000 as duration_us
FROM ext_graph
"""

GRAPH_KERNEL_SQL = """
SELECT graphExec, sequence, kernelName, gridX, gridY, gridZ,
       workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize
FROM graphKernel WHERE graphExec = ?
"""

GRAPH_LAUNCH_SQL = """
SELECT pid, tid, apiName, graphExec, stream, gpuId, queueId,
       apiDuration_usec, opDuration_usec, gpuTime_usec
FROM graphLaunch WHERE graphExec = ?
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not db.table_exists("ext_graph"):
        return html.Div([html.H2("Graphs"), html.P("No graph data in this trace.")])

    try:
        df = db.query_df(GRAPH_SUMMARY_SQL)

        return html.Div([
            html.H2("Graph Captures"),
            dag.AgGrid(
                id="graph-summary-grid",
                rowData=df.to_dict("records"),
                columnDefs=[
                    {"field": "graph", "headerName": "Graph"},
                    {"field": "graphExec", "headerName": "GraphExec"},
                    {"field": "stream", "headerName": "Stream"},
                    {"field": "start", "headerName": "Start"},
                    {"field": "end", "headerName": "End"},
                    {"field": "duration_us", "headerName": "Duration (us)", "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32, "rowSelection": {"mode": "singleRow"}},
                style={"height": "300px"},
            ),
            html.Div(
                [
                    html.Button("Show Kernels", id="btn-kernels", n_clicks=0, style={"marginRight": "10px"}),
                    html.Button("Show Launches", id="btn-launches", n_clicks=0),
                ],
                style={"margin": "15px 0"},
            ),
            html.Div(id="graph-detail"),
        ])
    except Exception as e:
        return html.Div(f"Error loading graphs: {e}")


@callback(
    Output("graph-detail", "children"),
    Input("btn-kernels", "n_clicks"),
    Input("btn-launches", "n_clicks"),
    State("graph-summary-grid", "selectedRows"),
    prevent_initial_call=True,
)
def show_detail(k_clicks, l_clicks, selected):
    if not selected:
        return html.P("Select a row first.")

    graph_exec = selected[0]["graphExec"]
    ctx = dash.ctx.triggered_id

    if ctx == "btn-kernels" and db.table_exists("graphKernel"):
        df = db.query_df(GRAPH_KERNEL_SQL, params=(graph_exec,))
        if df.empty:
            return html.P(f"No kernels for graphExec {graph_exec}.")
        return html.Div([
            html.H3(f"Kernels for GraphExec {graph_exec}"),
            dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[{"field": c} for c in df.columns],
                defaultColDef={"sortable": True, "resizable": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "400px"},
            ),
        ])

    if ctx == "btn-launches" and db.table_exists("graphLaunch"):
        df = db.query_df(GRAPH_LAUNCH_SQL, params=(graph_exec,))
        if df.empty:
            return html.P(f"No launches for graphExec {graph_exec}.")
        return html.Div([
            html.H3(f"Launches for GraphExec {graph_exec}"),
            dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[{"field": c} for c in df.columns],
                defaultColDef={"sortable": True, "resizable": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "400px"},
            ),
        ])

    return html.P("Detail view not available for this trace.")
