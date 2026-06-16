import dash
from dash import html

from rpd_dash.util import db
from rpd_dash.util.html_table import make_table

dash.register_page(__name__, path="/metadata", name="Metadata")


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        df = db.query_df("SELECT tag, value FROM rocpd_metadata")

        if df.empty:
            return html.Div([html.H2("Metadata"), html.P("No metadata found.")])

        return html.Div([
            html.H2("Metadata"),
            make_table(
                columns=[
                    {"field": "tag", "header": "Tag"},
                    {"field": "value", "header": "Value"},
                ],
                rows=df.to_dict("records"),
                col_styles={"value": {"fontFamily": "monospace"}},
            ),
        ])
    except Exception as e:
        return html.Div(f"Error loading metadata: {e}")
