import dash
from dash import html
import dash_ag_grid as dag
import plotly.express as px
from dash import dcc
import pandas as pd

from rpd_dash.util import db

dash.register_page(__name__, path="/tl/gpu-timeline", name="GPU Timeline")

KERNEL_CAT_PATTERNS = {
    "communication": ["nccl", "rccl", "allreduce", "allgather", "reducescatter", "all_to_all", "broadcast"],
    "memcpy": ["CopyHostToDevice", "CopyDeviceToHost", "CopyDeviceToDevice", "FillBuffer"],
}


def _categorize_op(op_type, description):
    if op_type in ("Memcpy", "CopyHostToDevice", "CopyDeviceToHost", "CopyDeviceToDevice", "FillBuffer"):
        return "memcpy"
    if op_type == "Barrier":
        return "barrier"
    name = description if description else op_type
    lower = name.lower()
    for cat, patterns in KERNEL_CAT_PATTERNS.items():
        if any(p.lower() in lower for p in patterns):
            return cat
    return "computation"


def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _total_time(intervals):
    return sum(e - s for s, e in intervals)


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    try:
        df = db.query_df("""
            SELECT gpuId, B.string as opType, A.string as description,
                   rocpd_op.start, rocpd_op.end
            FROM rocpd_op
            JOIN rocpd_string A ON A.id = rocpd_op.description_id
            JOIN rocpd_string B ON B.id = rocpd_op.opType_id
        """)

        if df.empty:
            return html.Div([html.H2("GPU Timeline"), html.P("No GPU operations found.")])

        df["category"] = df.apply(lambda r: _categorize_op(r["opType"], r["description"]), axis=1)

        gpu_ids = sorted(df["gpuId"].unique())
        multi_gpu = len(gpu_ids) > 1

        sections = [html.H2("GPU Timeline")]

        # Aggregate timeline
        agg_rows, agg_chart = _compute_timeline(df)
        sections.append(html.H3("Aggregate" if multi_gpu else ""))
        sections.append(dcc.Graph(figure=_make_pie(agg_chart, "GPU Time Breakdown")))
        sections.append(_make_table(agg_rows))

        # Per-GPU timelines
        if multi_gpu:
            per_gpu_summary = []
            for gpu_id in gpu_ids:
                gpu_df = df[df["gpuId"] == gpu_id]
                gpu_rows, gpu_chart = _compute_timeline(gpu_df)
                per_gpu_summary.extend([
                    {"GPU": gpu_id, **r} for r in gpu_rows
                ])
                sections.append(html.H3(f"GPU {gpu_id}", style={"marginTop": "25px"}))
                sections.append(dcc.Graph(figure=_make_pie(gpu_chart, f"GPU {gpu_id}")))
                sections.append(_make_table(gpu_rows))

        return html.Div(sections)
    except Exception as e:
        return html.Div(f"Error loading GPU timeline: {e}")


def _compute_timeline(df):
    categories = {"computation": [], "communication": [], "memcpy": [], "barrier": []}
    active_intervals = []

    for _, row in df.iterrows():
        interval = (row["start"], row["end"])
        cat = row["category"]
        if cat in categories:
            categories[cat].append(interval)
        if cat != "barrier":
            active_intervals.append(interval)

    merged = {cat: _merge_intervals(ivs) for cat, ivs in categories.items()}
    merged_active = _merge_intervals(active_intervals)

    total_wall = df["end"].max() - df["start"].min()
    if total_wall <= 0:
        return [], []

    busy_time = _total_time(merged_active)
    idle_time = total_wall - busy_time
    comp_time = _total_time(merged["computation"])
    comm_time = _total_time(merged["communication"])
    memcpy_time = _total_time(merged["memcpy"])
    barrier_time = _total_time(merged["barrier"])

    exposed_comm = _total_time(_subtract_intervals(merged["communication"], merged["computation"]))
    exposed_memcpy = _total_time(_subtract_intervals(merged["memcpy"], merged["computation"] + merged["communication"]))
    exposed_barrier = _total_time(_subtract_intervals(merged["barrier"], merged["computation"] + merged["communication"] + merged["memcpy"]))

    rows = [
        {"type": "Computation", "time_us": comp_time / 1000, "pct": comp_time * 100.0 / total_wall},
        {"type": "Communication (total)", "time_us": comm_time / 1000, "pct": comm_time * 100.0 / total_wall},
        {"type": "Communication (exposed)", "time_us": exposed_comm / 1000, "pct": exposed_comm * 100.0 / total_wall},
        {"type": "MemCopy (total)", "time_us": memcpy_time / 1000, "pct": memcpy_time * 100.0 / total_wall},
        {"type": "MemCopy (exposed)", "time_us": exposed_memcpy / 1000, "pct": exposed_memcpy * 100.0 / total_wall},
    ]
    if barrier_time > 0:
        rows.append({"type": "Barrier (total)", "time_us": barrier_time / 1000, "pct": barrier_time * 100.0 / total_wall})
        rows.append({"type": "Barrier (exposed)", "time_us": exposed_barrier / 1000, "pct": exposed_barrier * 100.0 / total_wall})
    rows.extend([
        {"type": "Busy", "time_us": busy_time / 1000, "pct": busy_time * 100.0 / total_wall},
        {"type": "Idle", "time_us": idle_time / 1000, "pct": idle_time * 100.0 / total_wall},
        {"type": "Total Wall", "time_us": total_wall / 1000, "pct": 100.0},
    ])

    chart_types = {"Computation", "Communication (exposed)", "MemCopy (exposed)", "Barrier (exposed)", "Idle"}
    chart_data = [r for r in rows if r["type"] in chart_types and r["time_us"] > 0]
    return rows, chart_data


def _make_pie(chart_data, title):
    if not chart_data:
        return {}
    chart_df = pd.DataFrame(chart_data)
    fig = px.pie(chart_df, values="time_us", names="type", title=title, hole=0.4)
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    return fig


def _make_table(rows):
    tl_df = pd.DataFrame(rows)
    return dag.AgGrid(
        rowData=tl_df.to_dict("records"),
        columnDefs=[
            {"field": "type", "headerName": "Category", "flex": 2},
            {"field": "time_us", "headerName": "Time (us)", "flex": 1,
             "valueFormatter": {"function": "d3.format(',.0f')(params.value)"}},
            {"field": "pct", "headerName": "%", "flex": 1,
             "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
        ],
        defaultColDef={"sortable": True, "resizable": True},
        dashGridOptions={"rowHeight": 28, "headerHeight": 32},
        style={"height": "350px"},
    )


def _subtract_intervals(a_intervals, b_intervals):
    if not a_intervals:
        return []
    b_merged = _merge_intervals(list(b_intervals))
    result = []
    for a_start, a_end in _merge_intervals(list(a_intervals)):
        current = a_start
        for b_start, b_end in b_merged:
            if b_end <= current:
                continue
            if b_start >= a_end:
                break
            if b_start > current:
                result.append((current, b_start))
            current = max(current, b_end)
        if current < a_end:
            result.append((current, a_end))
    return result
