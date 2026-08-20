import dash
from dash import html

from rpd_dash.util import db

dash.register_page(__name__, path="/file-info", name="File Info")


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    return html.Div([
        html.H2("File Info"),
        html.Div(
            html.Div(className="skeleton-card", style={"height": "220px", "marginBottom": "25px"}),
            id="file-info-content",
            className="htmx-fade",
            **{"data-hx-get": "/api/page/file-info", "data-hx-trigger": "load", "data-hx-swap": "innerHTML"},
        ),
    ])
