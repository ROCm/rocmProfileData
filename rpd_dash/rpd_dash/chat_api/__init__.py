import json
import os
import queue
import sqlite3
import threading
import time
import uuid

from flask import Blueprint, Response, request

from rpd_dash.util import db

chat_bp = Blueprint("chat_api", __name__, url_prefix="/api/chat")

ACTIVE_SESSIONS: dict[str, dict] = {}

QUERY_TIMEOUT_S = 10
ROW_LIMIT = 500
MAX_TOOL_RESULTS_CHARS = 6000
MAX_TURNS = int(os.environ.get("RPD_CHAT_MAX_TURNS", "20"))
CHAT_TIMEOUT_S = int(os.environ.get("RPD_CHAT_TIMEOUT", "120"))
RPD_INFO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "RPD_INFO.md"
)


def _read_rpd_info() -> str:
    try:
        with open(RPD_INFO_PATH) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _get_client():
    from openai import OpenAI

    base_url = os.environ.get("RPD_CHAT_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("RPD_CHAT_MODEL", "default-model")
    api_key = os.environ.get("RPD_CHAT_API_KEY", "not-needed")
    return OpenAI(base_url=base_url, api_key=api_key), model


def _run_sql(sql: str) -> str:
    conn = sqlite3.connect(
        f"file:{db.rpd_path}?mode=ro", uri=True, timeout=QUERY_TIMEOUT_S
    )
    conn.execute("PRAGMA query_only = ON")
    try:
        cur = conn.execute(sql)
        if cur.description is None:
            return "Query executed with no results returned."
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(ROW_LIMIT + 1)
        truncated = len(rows) > ROW_LIMIT
        if truncated:
            rows = rows[:ROW_LIMIT]
        if not rows:
            return "Query returned 0 rows."
        lines = [", ".join(str(c) for c in columns)]
        for row in rows:
            lines.append(", ".join(str(v) for v in row))
        header = f"(showing first {ROW_LIMIT})\n" if truncated else f"{len(rows)} rows\n"
        result = header + "\n".join(lines)
        if len(result) > MAX_TOOL_RESULTS_CHARS:
            result = result[:MAX_TOOL_RESULTS_CHARS] + "\n...(truncated)"
        return result
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()


def _chat_worker(
    session_id: str, stop_evt: threading.Event, messages: list, q: "queue.Queue"
):
    import json
    import time

    rpd_info = _read_rpd_info()
    client, model = _get_client()

    system_prompt = (
        "You are an assistant that analyzes RPD (ROCm Profile Data) GPU trace files. "
        "You have access to a read-only SQLite database containing the trace data.\n\n"
        "You can use the 'run_sql' tool to query the database. Iterate: query, analyze, "
        "and run follow-up queries as needed before providing a final comprehensive answer.\n\n"
        "Format your responses in markdown. Use tables for structured data and "
        "bullet points for lists. Be specific with numbers from the trace.\n"
    )
    if rpd_info:
        system_prompt += f"\n\n## Database Schema Reference\n{rpd_info}\n"

    conv = [{"role": "system", "content": system_prompt}] + messages

    tool_def = {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a read-only SQL query on the RPD trace database. "
                "Returns column headers and up to 500 rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute (read-only).",
                    }
                },
                "required": ["sql"],
            },
        },
    }

    def put(ev: dict):
        q.put(ev, block=False)

    try:
        put({"type": "info", "text": f"Schema: {len(rpd_info)} chars loaded"})
        turn = 0
        while turn < MAX_TURNS and not stop_evt.is_set():
            turn += 1
            put({"type": "progress", "text": f"[turn {turn}/{MAX_TURNS}] Calling model..."})
            resp = client.chat.completions.create(
                model=model, messages=conv, tools=[tool_def]
            )
            choice = resp.choices[0]

            if choice.message.tool_calls:
                conv.append(choice.message.model_dump())
                for tc in choice.message.tool_calls:
                    if tc.function.name == "run_sql":
                        sql = json.loads(tc.function.arguments)["sql"]
                        put({"type": "sql", "text": sql[:120]})
                        t0 = time.time()
                        result = _run_sql(sql)
                        elapsed = time.time() - t0
                        put({"type": "progress", "text": f"  -> {len(result)} chars in {elapsed:.2f}s"})
                        conv.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tc.id,
                        })
                continue

            if choice.message.content:
                put({"type": "progress", "text": f"Done ({turn} turns)."})
                put({"type": "answer", "content": choice.message.content})
                put({"type": "done", "status": "done"})
                return

        if stop_evt.is_set():
            put({"type": "done", "status": "cancelled"})
        else:
            put({
                "type": "answer",
                "content": f"Maximum turns reached ({MAX_TURNS}). Analysis incomplete.",
            })
            put({"type": "done", "status": "done"})

    except Exception as e:
        put({"type": "error", "message": str(e)})
        put({"type": "done", "status": "error"})
    finally:
        q.put(None)  # sentinel
        ACTIVE_SESSIONS.pop(session_id, None)


def _sse_stream(session_id: str, stop_evt: threading.Event, messages: list):
    q: "queue.Queue" = queue.Queue()
    t = threading.Thread(
        target=_chat_worker, args=(session_id, stop_evt, messages, q), daemon=True
    )
    t.start()

    yield f"data: {json.dumps({'type': 'start', 'sessionId': session_id})}\n\n"

    import queue as qmod

    while True:
        try:
            ev = q.get(timeout=1)
            if ev is None:
                break
            yield f"data: {json.dumps(ev)}\n\n"
        except qmod.Empty:
            yield ":\n\n"
            if not t.is_alive():
                break

    yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"


@chat_bp.route("/send", methods=["POST"])
def send():
    if not db.rpd_path:
        return {"error": "No RPD file loaded"}, 400

    data = request.get_json()
    user_text = data.get("text", "").strip()
    history = data.get("history", [])

    if not user_text:
        return {"error": "Empty message"}, 400

    session_id = str(uuid.uuid4())[:8]
    stop_evt = threading.Event()
    messages = list(history) + [{"role": "user", "content": user_text}]
    ACTIVE_SESSIONS[session_id] = {"stop": stop_evt, "messages": messages}

    return Response(
        _sse_stream(session_id, stop_evt, messages),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@chat_bp.route("/cancel/<session_id>", methods=["POST"])
def cancel(session_id: str):
    session = ACTIVE_SESSIONS.get(session_id)
    if session:
        session["stop"].set()
    return {"status": "ok"}
