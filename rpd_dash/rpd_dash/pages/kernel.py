import dash
from dash import html
import dash_ag_grid as dag

from rpd_dash.util import db

dash.register_page(__name__, path="/kernel", name="Kernels")

TOP_SQL = """
SELECT C.string as Name, count(C.string) as TotalCalls,
    sum(A.end - A.start) / 1000 as TotalDuration,
    (sum(A.end - A.start) / count(C.string)) / 1000 as Ave,
    sum(A.end - A.start) * 100.0 / (SELECT sum(A.end - A.start) FROM rocpd_op A) as Percentage
FROM (
    SELECT opType_id as name_id, start, end FROM rocpd_op
        WHERE description_id IN (SELECT id FROM rocpd_string WHERE string = '')
    UNION
    SELECT description_id, start, end FROM rocpd_op
        WHERE description_id NOT IN (SELECT id FROM rocpd_string WHERE string = '')
) A
JOIN rocpd_string C ON C.id = A.name_id
GROUP BY Name ORDER BY TotalDuration DESC
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        if db.table_exists("top"):
            df = db.query_df("SELECT * FROM top")
        else:
            df = db.query_df(TOP_SQL)

        df = df[df["Name"] != "Barrier"]

        dur_col = next(c for c in df.columns if c.lower().startswith("totalduration"))
        total_dur = df[dur_col].sum()
        if total_dur > 0:
            df["Percentage"] = df[dur_col] * 100.0 / total_dur
        ave_col = next(c for c in df.columns if c.lower().startswith("ave"))

        return html.Div([
            html.H2("Kernel Summary"),
            dag.AgGrid(
                rowData=df.to_dict("records"),
                columnDefs=[
                    {"field": "Name", "headerName": "Kernel", "flex": 3, "tooltipField": "Name"},
                    {"field": "TotalCalls", "headerName": "Calls", "flex": 1, "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                    {"field": dur_col, "headerName": "Total (us)", "flex": 1, "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                    {"field": ave_col, "headerName": "Avg (us)", "flex": 1, "valueFormatter": {"function": "d3.format(',')(params.value)"}},
                    {"field": "Percentage", "headerName": "%", "flex": 1, "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                ],
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "600px"},
            ),
        ])
    except Exception as e:
        return html.Div(f"Error loading kernel summary: {e}")
