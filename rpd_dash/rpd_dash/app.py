import sys
import os
import io
import gzip
import argparse

import dash
from dash import html, dcc, Input, Output, State, callback
from flask import request, Response

from rpd_dash.util import db

_pkg_dir = os.path.dirname(__file__)
_pages_dir = os.path.join(_pkg_dir, "pages")


def _find_rocpd_util():
    """Locate rocpd/util for chrometracing import."""
    candidates = [
        os.path.join(_pkg_dir, "..", "..", "rocpd", "util"),
        os.path.join(sys.prefix, "lib", "rocpd", "util"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            sys.path.insert(0, os.path.abspath(c))
            return
    try:
        import rocpd
        if getattr(rocpd, "__file__", None) is not None:
            p = os.path.join(os.path.dirname(rocpd.__file__), "..", "rocpd", "util")
            if os.path.isdir(p):
                sys.path.insert(0, os.path.abspath(p))
    except ImportError:
        pass


_find_rocpd_util()


def _parse_args():
    parser = argparse.ArgumentParser(description="Interactive viewer for RPD trace files")
    parser.add_argument("rpd_file", nargs="?", default=None, help="Path to .rpd trace file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to (default: 8050)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    return parser.parse_args()


def _create_app():
    args = _parse_args()

    if args.rpd_file:
        if not os.path.isfile(args.rpd_file):
            print(f"Error: file not found: {args.rpd_file}")
            sys.exit(1)
        db.set_rpd_path(os.path.abspath(args.rpd_file))
        print(f"Loaded: {db.rpd_path}")

    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder=_pages_dir,
        suppress_callback_exceptions=True,
    )

    NAV_LINKS = [
        ("Dashboard", "/"),
        ("Kernels", "/kernel"),
        ("API Calls", "/api"),
        ("GPU Ops", "/op"),
        ("Copies", "/copy"),
        ("Timeline", "/trace"),
        ("Monitor", "/monitor"),
        ("Graphs", "/graphs"),
        ("Autograd", "/autograd"),
        ("Metadata", "/metadata"),
        ("Counters", "/counters"),
        ("SQL Query", "/query"),
    ]

    TL_LINKS = [
        ("GPU Timeline", "/tl/gpu-timeline"),
        ("Kernel Categories", "/tl/kernel-categories"),
        ("Short Kernels", "/tl/short-kernels"),
        ("Torch Ops", "/tl/torch-ops"),
        ("Ops by Category", "/tl/ops-by-category"),
    ]

    link_style = {
        "display": "block",
        "padding": "8px 12px",
        "marginBottom": "4px",
        "textDecoration": "none",
        "color": "#ddd",
        "borderRadius": "4px",
    }

    sidebar = html.Div(
        [
            html.H2("RPD Viewer", style={"marginBottom": "20px", "color": "#ddd"}),
            html.Hr(),
            dcc.Link(id="rpd-filename", href="/file-info",
                     style={"fontSize": "12px", "color": "#aaa", "marginBottom": "15px", "display": "block", "textDecoration": "none"}),
            html.Nav([
                dcc.Link(label, href=href, className="nav-link", style=link_style)
                for label, href in NAV_LINKS
            ]),
            html.Div(
                [
                    html.Div("Analysis", style={
                        "fontSize": "11px", "color": "#888", "textTransform": "uppercase",
                        "letterSpacing": "1px", "padding": "12px 12px 4px",
                    }),
                    html.Nav([
                        dcc.Link(label, href=href, className="nav-link", style=link_style)
                        for label, href in TL_LINKS
                    ]),
                ],
                style={"borderTop": "1px solid #444", "marginTop": "10px"},
            ),
        ],
        style={
            "width": "220px",
            "minHeight": "100vh",
            "backgroundColor": "#2c2c2c",
            "padding": "20px",
            "position": "fixed",
            "top": 0,
            "left": 0,
        },
    )

    file_picker = html.Div(
        [
            html.H2("Load RPD File"),
            html.P("Enter the path to an .rpd trace file:"),
            dcc.Input(
                id="file-path-input",
                type="text",
                placeholder="/path/to/trace.rpd",
                style={"width": "500px", "padding": "8px", "fontSize": "14px"},
            ),
            html.Button(
                "Load",
                id="load-btn",
                n_clicks=0,
                style={"marginLeft": "10px", "padding": "8px 20px", "fontSize": "14px"},
            ),
            html.Div(id="file-error", style={"color": "red", "marginTop": "10px"}),
        ],
        style={"padding": "60px", "textAlign": "center"},
    )

    app.layout = html.Div([
        dcc.Store(id="rpd-loaded", data=db.rpd_path is not None),
        html.Div(
            id="main-app",
            children=[
                sidebar,
                html.Div(
                    dash.page_container,
                    style={"marginLeft": "260px", "padding": "20px", "flex": "1",
                           "minWidth": "0", "overflow": "hidden"},
                ),
            ],
            style={"display": "flex" if db.rpd_path else "none"},
        ),
        html.Div(
            id="picker-container",
            children=file_picker,
            style={"display": "none" if db.rpd_path else "block"},
        ),
    ])

    @callback(
        Output("rpd-filename", "children"),
        Input("rpd-loaded", "data"),
    )
    def show_filename(_):
        if db.rpd_path:
            return os.path.basename(db.rpd_path)
        return ""

    @callback(
        Output("main-app", "style"),
        Output("picker-container", "style"),
        Output("rpd-loaded", "data"),
        Output("file-error", "children"),
        Input("load-btn", "n_clicks"),
        State("file-path-input", "value"),
        prevent_initial_call=True,
    )
    def load_file(n_clicks, path):
        if not path or not path.strip():
            return dash.no_update, dash.no_update, dash.no_update, "Please enter a file path."
        path = path.strip()
        if not os.path.isfile(path):
            return dash.no_update, dash.no_update, dash.no_update, f"File not found: {path}"
        db.set_rpd_path(path)
        return {"display": "flex"}, {"display": "none"}, True, ""

    server = app.server

    @server.route("/tracedata")
    def serve_trace_json():
        from chrometracing import generateJson

        trace_args = argparse.Namespace()
        trace_args.input_rpd = db.rpd_path
        trace_args.format = "object"
        trace_args.start = "0%"
        trace_args.end = "100%"

        mem = io.StringIO()
        generateJson(mem, trace_args)
        data = mem.getvalue().encode("utf-8")

        headers = {"Content-Disposition": "attachment; filename=trace.json"}

        if "gzip" in request.headers.get("Accept-Encoding", ""):
            data = gzip.compress(data)
            headers["Content-Encoding"] = "gzip"

        return Response(data, mimetype="application/json", headers=headers)

    @server.route("/download-rpd")
    def download_rpd():
        if not db.rpd_path or not os.path.isfile(db.rpd_path):
            return Response("No RPD file loaded", status=404)

        filename = os.path.basename(db.rpd_path)
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

        if "gzip" in request.headers.get("Accept-Encoding", ""):
            import zlib
            import struct

            def generate():
                yield b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff'
                crc = zlib.crc32(b"")
                size = 0
                compress = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
                with open(db.rpd_path, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        crc = zlib.crc32(chunk, crc)
                        size += len(chunk)
                        compressed = compress.compress(chunk)
                        if compressed:
                            yield compressed
                yield compress.flush()
                yield struct.pack("<II", crc & 0xFFFFFFFF, size & 0xFFFFFFFF)

            headers["Content-Encoding"] = "gzip"
            return Response(generate(), mimetype="application/octet-stream", headers=headers)
        else:
            def generate():
                with open(db.rpd_path, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk

            return Response(generate(), mimetype="application/octet-stream", headers=headers)

    return app, args


def main():
    app, args = _create_app()
    app.run(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == "__main__":
    main()
