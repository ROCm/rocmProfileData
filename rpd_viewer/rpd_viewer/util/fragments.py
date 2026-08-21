"""HTML fragment renderers for htmx-loaded page sections.

These build raw HTML strings (not Dash components) because Flask routes
can't return Dash component trees directly -- htmx expects `text/html`
fragments that get swapped into the DOM.
"""
import re
from html import escape as _esc

from rpd_viewer.util import db
from rpd_viewer.util.html_table import render_table_html

BUSY_SQL = """
SELECT A.gpuId, GpuTime / 1000 as GpuTime_us, WallTime / 1000 as WallTime_us,
       GpuTime * 100.0 / WallTime as BusyPct
FROM (SELECT gpuId, sum(end - start) as GpuTime FROM rocpd_op
      WHERE opType_id NOT IN (SELECT id FROM rocpd_string WHERE string = 'Barrier')
      GROUP BY gpuId) A
INNER JOIN (SELECT max(end) - min(start) as WallTime FROM rocpd_op)
"""


def _stat_card_html(title, value):
    return (
        '<div style="padding:20px 30px;background-color:#f5f5f5;border-radius:8px;min-width:150px">'
        f'<div style="font-size:13px;color:#888">{_esc(title)}</div>'
        f'<div style="font-size:28px;font-weight:bold" '
        f'data-x-data="{{ target: {value!r}, display: \'0\' }}" '
        f'data-x-init="if (typeof target !== \'number\') {{ display = target }} '
        f'else {{ let s = 0; const step = () => {{ s += Math.ceil((target - s) / 8) || (target > s ? 1 : 0); '
        f"display = s.toLocaleString(); if (s < target) requestAnimationFrame(step); else display = target.toLocaleString() }}; step() }}\" "
        f'data-x-text="display">{_esc(str(value))}</div>'
        "</div>"
    )


def _dashboard_stats():
    conn = db.get_connection()
    try:
        api_calls = conn.execute("SELECT count(*) FROM rocpd_api").fetchone()[0]
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        row = conn.execute("SELECT MIN(start), MAX(end) FROM rocpd_api").fetchone()
        duration_s = (row[1] - row[0]) / 1e9 if row[0] and row[1] else 0
        gpus = conn.execute("SELECT count(DISTINCT gpuId) FROM rocpd_op").fetchone()[0]
    finally:
        conn.close()
    return {"api_calls": api_calls, "ops": ops, "duration_s": duration_s, "gpus": gpus}


def dashboard_stats_html():
    stats = _dashboard_stats()
    cards = "".join([
        _stat_card_html("API Calls", stats["api_calls"]),
        _stat_card_html("GPU Ops", stats["ops"]),
        _stat_card_html("GPUs", stats["gpus"]),
        _stat_card_html("Duration", f"{stats['duration_s']:.3f} s"),
    ])
    return f'<div style="display:flex;gap:20px" id="dash-stats-inner">{cards}</div>'


def dashboard_stats_sse_html():
    """Same stat cards, but without the count-up animation -- used for the
    periodic SSE live refresh so values just update in place."""
    stats = _dashboard_stats()

    def plain_card(title, value):
        return (
            '<div style="padding:20px 30px;background-color:#f5f5f5;border-radius:8px;min-width:150px">'
            f'<div style="font-size:13px;color:#888">{_esc(title)}</div>'
            f'<div style="font-size:28px;font-weight:bold">{_esc(str(value))}</div>'
            "</div>"
        )

    cards = "".join([
        plain_card("API Calls", f"{stats['api_calls']:,}"),
        plain_card("GPU Ops", f"{stats['ops']:,}"),
        plain_card("GPUs", stats["gpus"]),
        plain_card("Duration", f"{stats['duration_s']:.3f} s"),
    ])
    return f'<div style="display:flex;gap:20px" id="dash-stats-inner">{cards}</div>'


def dashboard_busy_html():
    busy_df = db.query_df(BUSY_SQL)
    if busy_df.empty:
        return ""

    table = render_table_html(
        columns=[
            {"field": "gpuId", "header": "GPU"},
            {"field": "GpuTime_us", "header": "GPU Time (us)", "format": lambda v: f"{int(v):,}"},
            {"field": "WallTime_us", "header": "Wall Time (us)", "format": lambda v: f"{int(v):,}"},
            {"field": "BusyPct", "header": "Busy %", "format": lambda v: f"{v:.1f}"},
        ],
        rows=busy_df.to_dict("records"),
        col_styles={"GpuTime_us": {"textAlign": "right"}, "WallTime_us": {"textAlign": "right"}, "BusyPct": {"textAlign": "right"}},
    )
    return f'<h3>GPU Utilization</h3>{table}'


def dashboard_domains_html():
    if not (db.has_annotations() and db.table_exists("api")):
        return ""

    parts = []
    domain_df = db.query_df(
        "SELECT domain, count(*) as calls FROM api GROUP BY domain ORDER BY calls DESC"
    )
    datasources = db.query_df(
        "SELECT tag, value FROM rocpd_metadata WHERE tag LIKE 'process_datasource%'"
    )

    if not domain_df.empty:
        table = render_table_html(
            columns=[
                {"field": "domain", "header": "Domain"},
                {"field": "calls", "header": "Calls", "format": lambda v: f"{int(v):,}"},
            ],
            rows=domain_df.to_dict("records"),
            col_styles={"calls": {"textAlign": "right"}},
        )
        parts.append(f'<h3 style="margin-top:25px">Trace Domains</h3>{table}')

    if not datasources.empty:
        sources = datasources["value"].apply(lambda v: re.sub(r".*source=", "", v))
        unique_sources = sorted(sources.unique())
        ds_rows = [{"source": s} for s in unique_sources]
        table = render_table_html(columns=[{"field": "source", "header": "Source"}], rows=ds_rows)
        parts.append(f'<h3 style="margin-top:15px">Data Sources</h3>{table}')

    return "".join(parts)


def metadata_html():
    df = db.query_df("SELECT tag, value FROM rocpd_metadata")
    if df.empty:
        return "<p>No metadata found.</p>"
    return render_table_html(
        columns=[
            {"field": "tag", "header": "Tag"},
            {"field": "value", "header": "Value"},
        ],
        rows=df.to_dict("records"),
        col_styles={"value": {"fontFamily": "monospace"}},
    )


COUNTER_SQL = """
SELECT counterName, dispatches, avg, min, max
FROM counter_summary
WHERE kernelName = ?
ORDER BY counterName
"""


def counter_panel_html(kernel):
    df = db.query_df(COUNTER_SQL, params=(kernel,))
    if df.empty:
        return "<p>No counter data.</p>"

    rows = df.to_dict("records")
    for r in rows:
        r["avg"] = round(r["avg"], 2)
        r["min"] = round(r["min"], 2)
        r["max"] = round(r["max"], 2)

    table = render_table_html(
        columns=[
            {"field": "counterName", "header": "Counter"},
            {"field": "dispatches", "header": "Dispatches", "format": lambda v: f"{int(v):,}"},
            {"field": "avg", "header": "Avg", "format": lambda v: f"{v:,.2f}"},
            {"field": "min", "header": "Min", "format": lambda v: f"{v:,.2f}"},
            {"field": "max", "header": "Max", "format": lambda v: f"{v:,.2f}"},
        ],
        rows=rows,
    )

    from urllib.parse import quote
    link = (
        f'<a href="/counters/detail?kernel={quote(kernel)}" '
        'style="color:#1a73e8;text-decoration:none;font-size:13px;display:inline-block;margin-top:10px">'
        "View all dispatches &rarr;</a>"
    )

    return table + link


def file_info_html():
    import os

    file_size = os.path.getsize(db.rpd_path)
    if file_size >= 1e9:
        size_str = f"{file_size / 1e9:.2f} GB"
    elif file_size >= 1e6:
        size_str = f"{file_size / 1e6:.2f} MB"
    else:
        size_str = f"{file_size / 1e3:.1f} KB"

    meta_df = db.query_df("SELECT tag, value FROM rocpd_metadata")

    info_block = (
        '<div style="padding:15px;background-color:#f5f5f5;border-radius:8px;margin-bottom:25px">'
        '<div style="margin-bottom:10px">'
        '<div style="font-size:13px;color:#888">File</div>'
        f'<div style="font-size:18px;font-weight:bold">{_esc(os.path.basename(db.rpd_path))}</div>'
        "</div>"
        '<div style="margin-bottom:10px">'
        '<div style="font-size:13px;color:#888">Path</div>'
        f'<div style="font-size:14px;font-family:monospace">{_esc(db.rpd_path)}</div>'
        "</div>"
        '<div style="margin-bottom:20px">'
        '<div style="font-size:13px;color:#888">Size</div>'
        f'<div style="font-size:18px">{_esc(size_str)}</div>'
        "</div>"
        "</div>"
        '<a href="/download-rpd" download="{}" '
        'style="display:inline-block;padding:12px 24px;font-size:14px;background-color:#0066cc;'
        'color:white;text-decoration:none;border-radius:4px;margin-bottom:25px">Download RPD File</a>'
    ).format(_esc(os.path.basename(db.rpd_path)))

    meta_block = ""
    if not meta_df.empty:
        table = render_table_html(
            columns=[
                {"field": "tag", "header": "Tag"},
                {"field": "value", "header": "Value"},
            ],
            rows=meta_df.to_dict("records"),
            col_styles={"value": {"fontFamily": "monospace"}},
        )
        meta_block = f"<h3>Metadata</h3>{table}"

    return info_block + meta_block
