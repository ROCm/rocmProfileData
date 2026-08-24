import json
import time
import dash
from dash import html, dcc
import dash_ag_grid as dag
import pandas as pd

from rpd_viewer.util import db

dash.register_page(__name__, path="/tl/torch-ops", name="Torch Ops")

TORCH_SHAPES_SQL = """
SELECT apiName, category, args,
       (end - start) / 1000 as cpu_time_us
FROM api
WHERE domain = 'torch' AND category IN ('function', 'backward_function')
AND apiName LIKE 'aten::%'
"""


def _build_torch_gpu_attribution(conn):
    conn.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS ext_torch (
            "id" integer NOT NULL PRIMARY KEY,
            "pid" integer NOT NULL, "tid" integer NOT NULL,
            "name" varchar(255) NOT NULL, "category" varchar(255) NOT NULL,
            "start" integer NOT NULL, "end" integer NOT NULL
        )
    """)
    conn.execute("DELETE FROM ext_torch")
    conn.execute("""
        INSERT INTO ext_torch (id, pid, tid, name, category, start, end)
        SELECT tmp_api.id, pid, tid, A.string, B.string, tmp_api.start, tmp_api.end
        FROM tmp_api
        JOIN rocpd_string A ON A.id = tmp_api.apiName_id
        JOIN rocpd_string C ON C.id = tmp_api.domain_id AND C.string = 'torch'
        JOIN rocpd_string B ON B.id = tmp_api.category_id AND B.string IN ('function', 'backward_function')
    """)

    conn.execute("DROP VIEW IF EXISTS torch_api")
    conn.execute("""
        CREATE TEMPORARY VIEW torch_api AS
        SELECT A.pid, A.tid, A.id as torch_id, A.name as torch_name, A.category, B.id as api_id
        FROM ext_torch A JOIN tmp_api B
        ON B.start BETWEEN A.start AND A.end AND A.pid = B.pid AND A.tid = B.tid
        WHERE torch_id IN (SELECT id FROM ext_torch)
    """)

    conn.execute("DROP VIEW IF EXISTS torch_kernel")
    conn.execute("""
        CREATE TEMPORARY VIEW torch_kernel AS
        SELECT torch_name, category, torch_id, gpuid, C.description_id, (C.end - C.start) as duration
        FROM tmp_api_ops A JOIN torch_api B ON B.api_id = A.api_id JOIN rocpd_op C ON C.id = A.op_id
        WHERE torch_id IN (SELECT id FROM ext_torch)
    """)


def _extract_shapes(args_str):
    try:
        data = json.loads(args_str)
        sizes = data.get("sizes", [])
        non_empty = [s for s in sizes if s]
        if non_empty:
            return str(non_empty)
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    if not db.has_torch_ops():
        return html.Div([html.H2("Torch Ops"), html.P("No PyTorch annotations in this trace.")])

    try:
        fmt_num = {"function": "d3.format(',')(params.value)"}
        fmt_dec = {"function": "d3.format(',.1f')(params.value)"}

        t0 = time.perf_counter()
        conn = db.get_indexed_connection()
        t_index = time.perf_counter() - t0

        t0 = time.perf_counter()
        _build_torch_gpu_attribution(conn)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        ops_agg = pd.read_sql_query("""
            SELECT torch_name as apiName, category,
                   COUNT(DISTINCT torch_id) as calls,
                   sum(duration) / 1000 as gpu_time_us
            FROM torch_kernel
            GROUP BY torch_name, category
            ORDER BY gpu_time_us DESC
        """, conn)
        t_query = time.perf_counter() - t0

        # Also get CPU time from a simple query
        cpu_df = db.query_df("""
            SELECT apiName, category, count(*) as calls,
                   sum(end - start) / 1000 as cpu_time_us,
                   avg(end - start) / 1000 as avg_cpu_us
            FROM api
            WHERE domain = 'torch' AND category IN ('function', 'backward_function')
            GROUP BY apiName, category
        """)

        ops_agg = ops_agg.merge(cpu_df[["apiName", "category", "cpu_time_us", "avg_cpu_us"]],
                                on=["apiName", "category"], how="left")

        timing_info = (
            f"Index setup: {t_index:.3f}s | "
            f"Attribution build: {t_build:.3f}s | "
            f"Query: {t_query:.3f}s | "
            f"Total: {t_index + t_build + t_query:.3f}s"
        )

        content = [
            html.Div(timing_info, style={"fontSize": "11px", "color": "#888", "marginBottom": "15px", "fontFamily": "monospace"}),
        ]

        col_defs = [
            {"field": "apiName", "headerName": "Op", "flex": 3},
            {"field": "calls", "headerName": "Calls", "flex": 1, "valueFormatter": fmt_num},
            {"field": "cpu_time_us", "headerName": "CPU Time (us)", "flex": 1, "valueFormatter": fmt_num},
            {"field": "avg_cpu_us", "headerName": "Avg CPU (us)", "flex": 1, "valueFormatter": fmt_dec},
            {"field": "gpu_time_us", "headerName": "GPU Time (us)", "flex": 1, "valueFormatter": fmt_num},
        ]

        fwd_df = ops_agg[ops_agg["category"] == "function"]
        bwd_df = ops_agg[ops_agg["category"] == "backward_function"]

        if not fwd_df.empty:
            content.append(html.H3("Forward Ops"))
            content.append(dag.AgGrid(
                rowData=fwd_df.to_dict("records"),
                columnDefs=col_defs,
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "400px"},
            ))

        if not bwd_df.empty:
            content.append(html.H3("Backward Ops", style={"marginTop": "25px"}))
            content.append(dag.AgGrid(
                rowData=bwd_df.to_dict("records"),
                columnDefs=col_defs,
                defaultColDef={"sortable": True, "resizable": True, "filter": True},
                dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                style={"height": "400px"},
            ))

        # Unique args / shapes view
        shapes_df = db.query_df(TORCH_SHAPES_SQL)
        if not shapes_df.empty:
            shapes_df["shapes"] = shapes_df["args"].apply(_extract_shapes)
            shaped = shapes_df[shapes_df["shapes"] != ""]
            if not shaped.empty:
                grouped = shaped.groupby(["apiName", "category", "shapes"]).agg(
                    calls=("cpu_time_us", "count"),
                    total_cpu_us=("cpu_time_us", "sum"),
                    avg_cpu_us=("cpu_time_us", "mean"),
                ).reset_index().sort_values("total_cpu_us", ascending=False)

                content.append(html.H3("Ops by Input Shape", style={"marginTop": "25px"}))
                content.append(dag.AgGrid(
                    rowData=grouped.to_dict("records"),
                    columnDefs=[
                        {"field": "apiName", "headerName": "Op", "flex": 2},
                        {"field": "category", "headerName": "Fwd/Bwd", "flex": 1},
                        {"field": "shapes", "headerName": "Input Shapes", "flex": 3, "tooltipField": "shapes"},
                        {"field": "calls", "headerName": "Calls", "flex": 1, "valueFormatter": fmt_num},
                        {"field": "total_cpu_us", "headerName": "Total CPU (us)", "flex": 1, "valueFormatter": fmt_num},
                        {"field": "avg_cpu_us", "headerName": "Avg CPU (us)", "flex": 1, "valueFormatter": fmt_dec},
                    ],
                    defaultColDef={"sortable": True, "resizable": True, "filter": True},
                    dashGridOptions={"rowHeight": 28, "headerHeight": 32},
                    style={"height": "500px"},
                ))

        return html.Div([
            html.H2("Torch Ops Summary"),
            dcc.Loading(type="circle", children=html.Div(content)),
        ])
    except Exception as e:
        return html.Div(f"Error loading torch ops: {e}")
