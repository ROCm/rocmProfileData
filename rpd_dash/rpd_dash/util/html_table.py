from dash import html


_HEADER_STYLE = {
    "textAlign": "left",
    "padding": "8px 14px",
    "fontSize": "12px",
    "color": "#888",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "borderBottom": "2px solid #e0e0e0",
}

_CELL_STYLE = {
    "padding": "8px 14px",
    "fontSize": "13px",
    "borderBottom": "1px solid #f0f0f0",
}

_ROW_ALT_BG = "#fafafa"


def make_table(columns, rows, col_styles=None):
    """Build a styled HTML table.

    columns: list of {"field": str, "header": str, "format": callable or None}
    rows:    list of dicts
    col_styles: optional dict of field -> extra style dict for that column
    """
    col_styles = col_styles or {}

    thead = html.Thead(html.Tr([
        html.Th(c["header"], style={
            **_HEADER_STYLE,
            **col_styles.get(c["field"], {}),
        }) for c in columns
    ]))

    tbody_rows = []
    for i, row in enumerate(rows):
        bg = {"backgroundColor": _ROW_ALT_BG} if i % 2 == 1 else {}
        cells = []
        for c in columns:
            val = row.get(c["field"], "")
            fmt = c.get("format")
            if fmt and val is not None and val != "":
                val = fmt(val)
            cells.append(html.Td(str(val), style={
                **_CELL_STYLE,
                **bg,
                **col_styles.get(c["field"], {}),
            }))
        tbody_rows.append(html.Tr(cells))

    return html.Table(
        [thead, html.Tbody(tbody_rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "borderRadius": "8px",
            "overflow": "hidden",
            "border": "1px solid #e0e0e0",
        },
    )
