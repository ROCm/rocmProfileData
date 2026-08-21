import dash
from dash import html, dcc
import dash_ag_grid as dag

from rpd_viewer.util import db

dash.register_page(__name__, path="/api", name="API Calls")

API_SQL = """
SELECT A.string as apiName, count(*) as total_calls
FROM rocpd_api
INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id
GROUP BY apiName
ORDER BY total_calls DESC
"""

API_ANNOTATED_SQL = """
SELECT domain, category, apiName, count(*) as total_calls,
       sum(end - start) / 1000 as total_cpu_us
FROM api
GROUP BY domain, category, apiName
ORDER BY domain, total_calls DESC
"""

DOMAIN_SUMMARY_SQL = """
SELECT domain, count(*) as total_calls,
       sum(end - start) / 1000 as total_cpu_us
FROM api
GROUP BY domain
ORDER BY total_calls DESC
"""

fmt_num = {"function": "d3.format(',')(params.value)"}


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        sections = [html.H2("API Call Summary")]

        if db.has_annotations() and db.table_exists("api"):
            domain_df = db.query_df(DOMAIN_SUMMARY_SQL)
            sections.append(html.H3("By Domain"))
            sections.append(dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=domain_df.to_dict("records"),
                columnDefs=[
                    {"field": "domain", "headerName": "Domain", "flex": 2},
                    {"field": "total_calls", "headerName": "Calls", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "total_cpu_us", "headerName": "CPU Time (us)", "flex": 1, "valueFormatter": fmt_num},
                ],
                defaultColDef={"sortable": True, "resizable": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "200px"},
            )))

            df = db.query_df(API_ANNOTATED_SQL)
            sections.append(html.H3("All API Calls", style={"marginTop": "25px"}))
            sections.append(dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[
                    {"field": "domain", "headerName": "Domain", "flex": 1},
                    {"field": "category", "headerName": "Category", "flex": 1},
                    {"field": "apiName", "headerName": "API Name", "flex": 3},
                    {"field": "total_calls", "headerName": "Calls", "flex": 1, "valueFormatter": fmt_num},
                    {"field": "total_cpu_us", "headerName": "CPU Time (us)", "flex": 1, "valueFormatter": fmt_num},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "600px"},
            )))
        else:
            if db.table_exists("api"):
                df = db.query_df("SELECT apiName, count(*) as total_calls FROM api GROUP BY apiName ORDER BY total_calls DESC")
            else:
                df = db.query_df(API_SQL)

            sections.append(dcc.Loading(type="circle", children=dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[
                    {"field": "apiName", "headerName": "API Name", "flex": 3},
                    {"field": "total_calls", "headerName": "Total Calls", "flex": 1, "valueFormatter": fmt_num},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "600px"},
            )))

        return html.Div(sections)
    except Exception as e:
        return html.Div(f"Error loading API summary: {e}")
