import os
import time

import dash
from dash import html, dcc

from rpd_dash.util import db

dash.register_page(__name__, path="/", name="Dashboard")


def _skeleton_cards():
    return html.Div(
        [
            html.Div(className="skeleton-card", style={"height": "78px", "minWidth": "150px", "flex": "1"})
            for _ in range(4)
        ],
        style={"display": "flex", "gap": "20px", "marginBottom": "30px"},
    )


def _is_live():
    """A trace file whose mtime changed in the last 30s is assumed to still
    be actively written to (e.g. a live profiling session)."""
    try:
        return (time.time() - os.path.getmtime(db.rpd_path)) < 30
    except OSError:
        return False


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    live = _is_live()

    stats_container = html.Div(
        _skeleton_cards(),
        id="dash-stats",
        className="htmx-fade",
        **{"data-hx-get": "/api/page/dashboard-stats", "data-hx-trigger": "load", "data-hx-swap": "innerHTML"},
    )

    if live:
        # Wrap the stats block with the htmx SSE extension so stat cards
        # refresh every ~2s while the trace file is still being written to.
        stats_container = html.Div(
            [
                html.Span(
                    "● LIVE",
                    style={
                        "fontSize": "11px", "color": "#1a9e5c", "fontWeight": "bold",
                        "marginBottom": "6px", "display": "inline-block",
                    },
                ),
                html.Div(
                    stats_container,
                    **{"data-sse-swap": "stats"},
                ),
            ],
            **{"data-hx-ext": "sse", "data-sse-connect": "/api/live-stats"},
        )

    return html.Div([
        html.H2("Dashboard"),
        stats_container,
        html.Div(
            html.Div(className="skeleton-card", style={"height": "200px", "marginBottom": "30px"}),
            id="dash-busy",
            className="htmx-fade",
            **{"data-hx-get": "/api/page/dashboard-busy", "data-hx-trigger": "load", "data-hx-swap": "innerHTML"},
        ),
        html.Div(
            id="dash-domains",
            className="htmx-fade",
            **{"data-hx-get": "/api/page/dashboard-domains", "data-hx-trigger": "load", "data-hx-swap": "innerHTML"},
        ),
    ])
