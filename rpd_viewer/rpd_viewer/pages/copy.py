import dash
from dash import html, dcc
import dash_ag_grid as dag

from rpd_viewer.util import db

dash.register_page(__name__, path="/copy", name="Copies")

COPY_SQL = """
SELECT E.string as apiName, count(*) as total_calls,
       sum(size) as total_bytes, avg(size) as avg_bytes,
       sum(O.end - O.start) / 1000 as total_duration_us,
       avg(O.end - O.start) / 1000 as avg_duration_us
FROM rocpd_api_ops AO
JOIN rocpd_op O ON O.id = AO.op_id
JOIN rocpd_copyapi C ON C.api_ptr_id = AO.api_id
JOIN rocpd_api D ON D.id = AO.api_id
JOIN rocpd_string E ON E.id = D.apiName_id
GROUP BY apiName
ORDER BY total_bytes DESC
"""

COPY_DETAIL_SQL = """
SELECT E.string as apiName, gpuId, size, dstDevice, srcDevice,
       sync, pinned, (O.end - O.start) / 1000 as duration_us,
       CASE WHEN (O.end - O.start) > 0
           THEN size * 1.0 / ((O.end - O.start) / 1000) / 1e3
           ELSE 0 END as bandwidth_gbs
FROM rocpd_api_ops AO
JOIN rocpd_op O ON O.id = AO.op_id
JOIN rocpd_copyapi C ON C.api_ptr_id = AO.api_id
JOIN rocpd_api D ON D.id = AO.api_id
JOIN rocpd_string E ON E.id = D.apiName_id
ORDER BY size DESC
LIMIT 1000
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not db.table_exists("rocpd_copyapi"):
        return html.Div([html.H2("Memory Copies"), html.P("No copy data in this trace.")])

    try:
        summary_df = db.query_df(COPY_SQL)

        if summary_df.empty:
            return html.Div([html.H2("Memory Copies"), html.P("No copy operations found.")])

        detail_df = db.query_df(COPY_DETAIL_SQL)

        fmt_num = {"function": "d3.format(',')(params.value)"}
        fmt_bytes = {"function": "params.value >= 1e9 ? d3.format('.2f')(params.value/1e9) + ' GB' : params.value >= 1e6 ? d3.format('.2f')(params.value/1e6) + ' MB' : params.value >= 1e3 ? d3.format('.2f')(params.value/1e3) + ' KB' : params.value + ' B'"}

        return html.Div([
            html.H2("Memory Copies"),
            html.H3("Summary by API"),
            dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=summary_df.to_dict("records"),
                columnDefs=[
                    {"field": "apiName", "headerName": "API", "flex": 2},
                    {"field": "total_calls", "headerName": "Calls", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "total_bytes", "headerName": "Total Size", "flex": 1, "valueFormatter": fmt_bytes},
                    {"field": "avg_bytes", "headerName": "Avg Size", "flex": 1, "valueFormatter": fmt_bytes},
                    {"field": "total_duration_us", "headerName": "Total (us)", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "avg_duration_us", "headerName": "Avg (us)", "flex": 1, "valueFormatter": fmt_num},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "250px"},
            )),
            html.H3("Individual Copies (top 1000 by size)", style={"marginTop": "30px"}),
            dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=detail_df.to_dict("records"),
                columnDefs=[
                    {"field": "apiName", "headerName": "API", "flex": 2},
                    {"field": "gpuId", "headerName": "GPU", "flex": 1},
                    {"field": "size", "headerName": "Size", "flex": 1, "valueFormatter": fmt_bytes},
                    {"field": "srcDevice", "headerName": "Src Device", "flex": 1},
                    {"field": "dstDevice", "headerName": "Dst Device", "flex": 1},
                    {"field": "sync", "headerName": "Sync", "flex": 1},
                    {"field": "pinned", "headerName": "Pinned", "flex": 1},
                    {"field": "duration_us", "headerName": "Duration (us)", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "bandwidth_gbs", "headerName": "BW (GB/s)", "flex": 1, "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "500px"},
            )),
        ])
    except Exception as e:
        return html.Div(f"Error loading copy summary: {e}")
