import dash
from dash import html, dcc

from rpd_dash.util import db

dash.register_page(__name__, path="/chat", name="Chat")


def layout():
    if not db.rpd_path:
        return html.Div("No RPD file loaded.")

    return html.Div(
        [
            html.H2("Chat"),
            html.P(
                "Ask questions about the loaded RPD trace file. "
                "The assistant can query the database to analyze your trace.",
                style={"color": "#888", "marginBottom": "20px"},
            ),
            html.Div(
                id="chat-spinner",
                style={
                    "display": "none",
                    "textAlign": "center",
                    "padding": "20px 0",
                    "color": "#1a73e8",
                },
                children="Thinking...",
            ),
            html.Div(
                id="chat-messages",
                style={
                    "minHeight": "100px",
                    "maxHeight": "calc(100vh - 340px)",
                    "overflowY": "auto",
                    "padding": "10px",
                },
            ),
            html.Div(
                id="chat-progress",
                style={
                    "display": "none",
                    "marginTop": "8px",
                    "padding": "8px 12px",
                    "backgroundColor": "#f5f5f5",
                    "borderRadius": "6px",
                    "border": "1px solid #e0e0e0",
                    "fontFamily": "monospace",
                    "fontSize": "12px",
                    "color": "#555",
                    "height": "120px",
                    "overflowY": "auto",
                    "whiteSpace": "pre-wrap",
                },
            ),
            html.Div(
                [
                    dcc.Textarea(
                        id="chat-input",
                        placeholder="Ask a question about this trace...",
                        disabled=False,
                        style={
                            "width": "100%",
                            "minHeight": "52px",
                            "maxHeight": "150px",
                            "resize": "vertical",
                            "padding": "10px 14px",
                            "fontSize": "14px",
                            "border": "1px solid #ccc",
                            "borderRadius": "8px",
                            "fontFamily": "inherit",
                            "outline": "none",
                            "boxSizing": "border-box",
                            "lineHeight": "1.4",
                        },
                    ),
                    html.Button(
                        "Send",
                        id="chat-send-btn",
                        style={
                            "padding": "10px 28px",
                            "fontSize": "14px",
                            "backgroundColor": "#1a73e8",
                            "color": "#fff",
                            "border": "none",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "Clear",
                        id="chat-clear-btn",
                        style={
                            "padding": "10px 20px",
                            "fontSize": "14px",
                            "backgroundColor": "#f5f5f5",
                            "color": "#666",
                            "border": "1px solid #ccc",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "marginLeft": "8px",
                        },
                    ),
                    html.Button(
                        "Cancel",
                        id="chat-cancel-btn",
                        style={
                            "padding": "10px 14px",
                            "fontSize": "13px",
                            "backgroundColor": "#f5f5f5",
                            "color": "#c00",
                            "border": "1px solid #ddd",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "marginLeft": "8px",
                            "display": "none",
                        },
                    ),
                    html.Button(
                        "Log",
                        id="chat-log-btn",
                        style={
                            "padding": "10px 14px",
                            "fontSize": "13px",
                            "backgroundColor": "#f5f5f5",
                            "color": "#666",
                            "border": "1px solid #ccc",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "marginLeft": "8px",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "10px", "marginTop": "12px"},
            ),
        ],
        style={"position": "relative"},
    )
