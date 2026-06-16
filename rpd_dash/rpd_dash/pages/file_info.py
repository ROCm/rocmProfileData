import os
import dash
from dash import html

from rpd_dash.util import db
from rpd_dash.util.html_table import make_table

dash.register_page(__name__, path="/file-info", name="File Info")


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        file_size = os.path.getsize(db.rpd_path)
        if file_size >= 1e9:
            size_str = f"{file_size / 1e9:.2f} GB"
        elif file_size >= 1e6:
            size_str = f"{file_size / 1e6:.2f} MB"
        else:
            size_str = f"{file_size / 1e3:.1f} KB"

        meta_df = db.query_df("SELECT tag, value FROM rocpd_metadata")

        sections = [
            html.H2("File Info"),
            html.Div([
                html.Div([
                    html.Div("File", style={"fontSize": "13px", "color": "#888"}),
                    html.Div(os.path.basename(db.rpd_path), style={"fontSize": "18px", "fontWeight": "bold"}),
                ], style={"marginBottom": "10px"}),
                html.Div([
                    html.Div("Path", style={"fontSize": "13px", "color": "#888"}),
                    html.Div(db.rpd_path, style={"fontSize": "14px", "fontFamily": "monospace"}),
                ], style={"marginBottom": "10px"}),
                html.Div([
                    html.Div("Size", style={"fontSize": "13px", "color": "#888"}),
                    html.Div(size_str, style={"fontSize": "18px"}),
                ], style={"marginBottom": "20px"}),
            ], style={"padding": "15px", "backgroundColor": "#f5f5f5", "borderRadius": "8px", "marginBottom": "25px"}),
            html.A(
                "Download RPD File",
                href="/download-rpd",
                download=os.path.basename(db.rpd_path),
                style={
                    "display": "inline-block",
                    "padding": "12px 24px",
                    "fontSize": "14px",
                    "backgroundColor": "#0066cc",
                    "color": "white",
                    "textDecoration": "none",
                    "borderRadius": "4px",
                    "marginBottom": "25px",
                },
            ),
        ]

        if not meta_df.empty:
            sections.append(html.H3("Metadata"))
            sections.append(make_table(
                columns=[
                    {"field": "tag", "header": "Tag"},
                    {"field": "value", "header": "Value"},
                ],
                rows=meta_df.to_dict("records"),
                col_styles={"value": {"fontFamily": "monospace"}},
            ))

        return html.Div(sections)
    except Exception as e:
        return html.Div(f"Error loading file info: {e}")
