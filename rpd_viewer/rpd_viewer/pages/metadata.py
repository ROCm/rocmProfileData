import dash
from dash import html

from rpd_viewer.util import db

dash.register_page(__name__, path="/metadata", name="Metadata")


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    return html.Div([
        html.H2("Metadata"),
        html.Div(
            html.Div(className="skeleton-card", style={"height": "300px"}),
            id="metadata-content",
            className="htmx-fade",
            **{"data-hx-get": "/api/page/metadata", "data-hx-trigger": "load", "data-hx-swap": "innerHTML"},
        ),
    ])
