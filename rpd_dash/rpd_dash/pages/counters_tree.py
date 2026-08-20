import dash
from dash import html

from rpd_dash.util import db

dash.register_page(__name__, path="/counters", name="Counters")

RANKING_SQL = """
SELECT C.string as kernelName, count(*) as TotalCalls,
    sum(A."end" - A.start) / 1000 as TotalDuration_us,
    (sum(A."end" - A.start) / count(*)) / 1000 as Avg_us,
    sum(A."end" - A.start) * 100.0 / (SELECT sum("end" - start) FROM rocpd_op) as Percentage
FROM rocpd_op A
JOIN rocpd_string C ON C.id = A.description_id
WHERE C.string IN (SELECT DISTINCT kernelName FROM counter_summary)
GROUP BY C.string
ORDER BY TotalDuration_us DESC
"""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not db.table_exists("rocpd_counter"):
        return html.Div([
            html.H2("GPU Counters"),
            html.P("No counter data in this trace."),
        ])

    try:
        df = db.query_df(RANKING_SQL)
        if df.empty:
            return html.Div([
                html.H2("GPU Counters"),
                html.P("No counter data in this trace."),
            ])

        rows = []
        for i, row in df.iterrows():
            kernel = row["kernelName"]

            header = html.Div([
                html.Span(
                    "▶",
                    style={
                        "flex": "0 0 20px",
                        "cursor": "pointer",
                        "fontSize": "12px",
                        "color": "#666",
                        "transition": "transform 0.15s",
                        "userSelect": "none",
                        "display": "inline-block",
                    },
                    **{"data-x-bind:style": "open ? 'transform:rotate(90deg)' : 'transform:rotate(0deg)'"},
                ),
                html.Span(kernel, title=kernel, style={
                    "flex": "1 1 0",
                    "minWidth": "0",
                    "fontFamily": "monospace",
                    "fontSize": "13px",
                    "fontWeight": "500",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                }),
                _stat("Calls", f"{int(row['TotalCalls']):,}", "80px"),
                _stat("Total", f"{int(row['TotalDuration_us']):,} us", "120px"),
                _stat("Avg", f"{row['Avg_us']:,.1f} us", "100px"),
                _stat("%", f"{row['Percentage']:.2f}", "60px"),
            ], style={
                   "padding": "10px 14px",
                   "cursor": "pointer",
                   "borderBottom": "1px solid #eee",
                   "display": "flex",
                   "alignItems": "center",
                   "overflow": "hidden",
               },
               **{"data-x-on:click": "open = !open"})

            detail_panel = html.Div(
                html.Div(className="skeleton-card", style={"height": "60px"}),
                className="htmx-fade",
                style={
                    "padding": "12px 14px 16px 36px",
                    "backgroundColor": "#fafafa",
                    "borderBottom": "1px solid #eee",
                },
                **{
                    "data-x-show": "open",
                    "data-x-transition": "",
                    "data-hx-get": f"/api/page/counter-detail?kernel={kernel}",
                    "data-hx-trigger": "intersect once",
                    "data-hx-swap": "innerHTML",
                },
            )

            rows.append(html.Div(
                [header, detail_panel],
                style={"overflow": "hidden"},
                **{"data-x-data": "{ open: false }"},
            ))

        return html.Div([
            html.H2("GPU Counters"),
            html.P(f"{len(df)} kernels with counter data, sorted by total GPU time",
                   style={"color": "#666", "marginBottom": "20px"}),
            html.Div(rows, style={
                "border": "1px solid #e0e0e0",
                "borderRadius": "8px",
                "backgroundColor": "#fff",
                "overflow": "hidden",
            }),
        ])
    except Exception as e:
        return html.Div(f"Error loading counters: {e}")


def _stat(label, value, width):
    return html.Span([
        html.Span(f"{label}: ", style={"color": "#999", "fontSize": "11px"}),
        html.Span(value, style={"fontSize": "12px"}),
    ], style={"flex": f"0 0 {width}", "textAlign": "right"})
