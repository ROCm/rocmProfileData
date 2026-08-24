import dash
from dash import html, dcc
import dash_ag_grid as dag

from rpd_viewer.util import db

dash.register_page(__name__, path="/autograd", name="Autograd")

AUTOGRAD_SQL = "SELECT autogradName, kernelName, sizes, calls, avg_gpu, total_gpu FROM autogradKernel"


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not db.table_exists("autogradKernel"):
        return html.Div([html.H2("Autograd"), html.P("No autograd data in this trace.")])

    try:
        df = db.query_df(AUTOGRAD_SQL)

        return html.Div([
            html.H2("Autograd Kernel Summary"),
            dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[
                    {"field": "autogradName", "headerName": "Autograd Operator", "flex": 2, "rowGroup": True, "hide": True},
                    {"field": "kernelName", "headerName": "Kernel", "flex": 3, "tooltipField": "kernelName"},
                    {"field": "sizes", "headerName": "Sizes", "flex": 1},
                    {"field": "calls", "headerName": "Calls", "flex": 1, "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                    {"field": "avg_gpu", "headerName": "Avg GPU (us)", "flex": 1, "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                    {"field": "total_gpu", "headerName": "Total GPU (us)", "flex": 1, "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32, 
                    "groupDefaultExpanded": 0,
                    "autoGroupColumnDef": {
                        "headerName": "Autograd Operator",
                        "minWidth": 300,
                        "cellRendererParams": {"suppressCount": False},
                    },
                },
                style={"height": "700px"},
            )),
        ])
    except Exception as e:
        return html.Div(f"Error loading autograd summary: {e}")
