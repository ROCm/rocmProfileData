import dash
from dash import html, dcc
import dash_ag_grid as dag

from rpd_viewer.util import db

dash.register_page(__name__, path="/kernel", name="Kernels")

def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        df = db.kernel_summary()

        df = df[df["Name"] != "Barrier"]

        dur_col = next(c for c in df.columns if c.lower().startswith("totalduration"))
        total_dur = df[dur_col].sum()
        if total_dur > 0:
            df["Percentage"] = df[dur_col] * 100.0 / total_dur
        ave_col = next(c for c in df.columns if c.lower().startswith("ave"))

        return html.Div([
            html.H2("Kernel Summary"),
            dcc.Loading(
                type="circle",
                children=dag.AgGrid(
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
            ),
        ])
    except Exception as e:
        return html.Div(f"Error loading kernel summary: {e}")
