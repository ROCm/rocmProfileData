import dash
from dash import html, Input, Output, ClientsideFunction

from rpd_dash.util import db

dash.register_page(__name__, path="/trace", name="Timeline")


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    return html.Div([
        html.H2("Timeline Trace"),
        html.P("Generate Chrome Tracing JSON and view in Perfetto or download."),
        html.Div(
            [
                html.Button("Open in Perfetto", id="btn-perfetto", n_clicks=0,
                             style={"marginRight": "10px", "padding": "10px 20px", "fontSize": "14px"}),
                html.A("Download JSON", href="/tracedata", download="trace.json",
                       style={"padding": "10px 20px", "fontSize": "14px", "textDecoration": "none",
                              "border": "1px solid #ccc", "borderRadius": "4px", "color": "#333"}),
            ],
            style={"margin": "20px 0"},
        ),
        html.Pre(id="trace-log", children="",
                 style={"border": "1px solid #eee", "padding": "10px",
                        "fontFamily": "monospace", "fontSize": "12px",
                        "minHeight": "60px", "marginTop": "15px"}),
    ])


dash.clientside_callback(
    ClientsideFunction(namespace="perfetto", function_name="open_trace"),
    Output("trace-log", "children"),
    Input("btn-perfetto", "n_clicks"),
    prevent_initial_call=True,
)
