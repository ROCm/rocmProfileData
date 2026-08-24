import dash
from dash import html, dcc
import dash_ag_grid as dag

from rpd_viewer.util import db

dash.register_page(__name__, path="/op", name="GPU Ops")

OP_SQL = """
SELECT B.string as opType, count(*) as total_execs
FROM rocpd_op
INNER JOIN rocpd_string B ON B.id = rocpd_op.opType_id
GROUP BY opType
ORDER BY total_execs DESC
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        if db.table_exists("op"):
            df = db.query_df("SELECT opType, count(*) as total_execs FROM op GROUP BY opType ORDER BY total_execs DESC")
        else:
            df = db.query_df(OP_SQL)

        return html.Div([
            html.H2("GPU Op Summary"),
            dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[
                    {"field": "opType", "headerName": "Op Type", "flex": 3},
                    {"field": "total_execs", "headerName": "Total Executions", "flex": 1, "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "600px"},
            )),
        ])
    except Exception as e:
        return html.Div(f"Error loading op summary: {e}")
