import dash
from dash import html, dcc, callback, Input, Output
import plotly.express as px

from rpd_dash.util import db
from rpd_dash.util.html_table import make_table

dash.register_page(__name__, path="/", name="Dashboard")

BUSY_SQL = """
SELECT A.gpuId, GpuTime / 1000 as GpuTime_us, WallTime / 1000 as WallTime_us,
       GpuTime * 100.0 / WallTime as BusyPct
FROM (SELECT gpuId, sum(end - start) as GpuTime FROM rocpd_op
      WHERE opType_id NOT IN (SELECT id FROM rocpd_string WHERE string = 'Barrier')
      GROUP BY gpuId) A
INNER JOIN (SELECT max(end) - min(start) as WallTime FROM rocpd_op)
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        stats = {}
        conn = db.get_connection()
        try:
            stats["api_calls"] = conn.execute("SELECT count(*) FROM rocpd_api").fetchone()[0]
            stats["ops"] = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
            row = conn.execute("SELECT MIN(start), MAX(end) FROM rocpd_api").fetchone()
            stats["min_ts"] = row[0]
            stats["max_ts"] = row[1]
            stats["duration_s"] = (row[1] - row[0]) / 1e9 if row[0] and row[1] else 0
            stats["gpus"] = conn.execute("SELECT count(DISTINCT gpuId) FROM rocpd_op").fetchone()[0]
        finally:
            conn.close()

        busy_df = db.query_df(BUSY_SQL)

        stat_cards = html.Div(
            [
                _card("API Calls", f"{stats['api_calls']:,}"),
                _card("GPU Ops", f"{stats['ops']:,}"),
                _card("GPUs", str(stats["gpus"])),
                _card("Duration", f"{stats['duration_s']:.3f} s"),
            ],
            style={"display": "flex", "gap": "20px", "marginBottom": "30px"},
        )

        busy_table = html.Div()
        if not busy_df.empty:
            busy_table = html.Div([
                html.H3("GPU Utilization"),
                make_table(
                    columns=[
                        {"field": "gpuId", "header": "GPU"},
                        {"field": "GpuTime_us", "header": "GPU Time (us)", "format": lambda v: f"{int(v):,}"},
                        {"field": "WallTime_us", "header": "Wall Time (us)", "format": lambda v: f"{int(v):,}"},
                        {"field": "BusyPct", "header": "Busy %", "format": lambda v: f"{v:.1f}"},
                    ],
                    rows=busy_df.to_dict("records"),
                    col_styles={"GpuTime_us": {"textAlign": "right"}, "WallTime_us": {"textAlign": "right"}, "BusyPct": {"textAlign": "right"}},
                ),
            ])

        sections = [html.H2("Dashboard"), stat_cards, busy_table]

        if db.has_annotations() and db.table_exists("api"):
            domain_df = db.query_df(
                "SELECT domain, count(*) as calls FROM api GROUP BY domain ORDER BY calls DESC"
            )
            datasources = db.query_df(
                "SELECT tag, value FROM rocpd_metadata WHERE tag LIKE 'process_datasource%'"
            )
            if not domain_df.empty:
                sections.append(html.H3("Trace Domains", style={"marginTop": "25px"}))
                sections.append(make_table(
                    columns=[
                        {"field": "domain", "header": "Domain"},
                        {"field": "calls", "header": "Calls", "format": lambda v: f"{int(v):,}"},
                    ],
                    rows=domain_df.to_dict("records"),
                    col_styles={"calls": {"textAlign": "right"}},
                ))
            if not datasources.empty:
                import re
                sources = datasources["value"].apply(lambda v: re.sub(r".*source=", "", v))
                unique_sources = sorted(sources.unique())
                ds_rows = [{"source": s} for s in unique_sources]
                sections.append(html.H3("Data Sources", style={"marginTop": "15px"}))
                sections.append(make_table(
                    columns=[{"field": "source", "header": "Source"}],
                    rows=ds_rows,
                ))

        return html.Div(sections)
    except Exception as e:
        return html.Div(f"Error loading dashboard: {e}")


def _card(title, value):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "13px", "color": "#888"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": "bold"}),
        ],
        style={
            "padding": "20px 30px",
            "backgroundColor": "#f5f5f5",
            "borderRadius": "8px",
            "minWidth": "150px",
        },
    )
