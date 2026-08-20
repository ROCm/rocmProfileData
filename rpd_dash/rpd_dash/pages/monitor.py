import dash
from dash import html, dcc
import plotly.express as px

from rpd_dash.util import db

dash.register_page(__name__, path="/monitor", name="GPU Monitor")

MONITOR_SQL = """
SELECT deviceId, monitorType, start / 1000000 as time_ms, CAST(value AS REAL) as value
FROM rocpd_monitor
ORDER BY start
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not db.table_exists("rocpd_monitor"):
        return html.Div([html.H2("GPU Monitor"), html.P("No monitor data in this trace.")])

    try:
        df = db.query_df(MONITOR_SQL)

        if df.empty:
            return html.Div([html.H2("GPU Monitor"), html.P("No monitor data found.")])

        min_time = df["time_ms"].min()
        df["time_ms"] = df["time_ms"] - min_time

        charts = []
        for monitor_type in df["monitorType"].unique():
            mdf = df[df["monitorType"] == monitor_type]
            unit = _unit(monitor_type)
            fig = px.line(
                mdf,
                x="time_ms",
                y="value",
                color="deviceId",
                title=f"{monitor_type}",
                labels={"time_ms": "Time (ms)", "value": unit, "deviceId": "GPU"},
            )
            fig.update_layout(height=300, margin=dict(t=40, b=30))
            charts.append(dcc.Graph(figure=fig))

        return html.Div([html.H2("GPU Monitor"), dcc.Loading(type="circle", children=html.Div(charts))])
    except Exception as e:
        return html.Div(f"Error loading monitor data: {e}")


def _unit(monitor_type):
    units = {
        "temp": "C",
        "power": "W",
        "sclk": "MHz",
        "mclk": "MHz",
        "gpu%": "%",
        "vram%": "%",
        "fan%": "%",
    }
    return units.get(monitor_type, "")
