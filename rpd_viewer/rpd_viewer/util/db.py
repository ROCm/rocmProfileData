import hashlib
import re
import threading

import sqlite3
import pandas as pd

rpd_path = None
_persistent_conn = None
_indexes_ready = False

_query_cache = {}
_query_cache_lock = threading.Lock()


def _apply_pragmas(conn):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-200000")


def get_connection():
    if rpd_path is None:
        raise RuntimeError("No RPD file loaded")
    conn = sqlite3.connect(rpd_path)
    _apply_pragmas(conn)
    return conn


def _get_persistent():
    global _persistent_conn, _indexes_ready
    if _persistent_conn is None or rpd_path is None:
        _indexes_ready = False
        if rpd_path is None:
            raise RuntimeError("No RPD file loaded")
        _persistent_conn = sqlite3.connect(rpd_path, check_same_thread=False)
        _apply_pragmas(_persistent_conn)
    return _persistent_conn


def _ensure_indexes_on(conn):
    conn.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS tmp_api AS
        SELECT id, pid, tid, start, end, apiName_id, domain_id, category_id
        FROM rocpd_api
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS tmp_api_tid_pid_start_idx ON tmp_api(tid,pid,start)")
    conn.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS tmp_api_ops AS
        SELECT api_id, op_id FROM rocpd_api_ops
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS tmp_api_ops_idx ON tmp_api_ops(api_id, op_id)")


def ensure_indexes():
    global _indexes_ready
    if _indexes_ready:
        return
    conn = _get_persistent()
    _ensure_indexes_on(conn)
    _indexes_ready = True


def get_indexed_connection():
    ensure_indexes()
    return _get_persistent()


def _normalize_sql(sql):
    return re.sub(r"\s+", " ", sql).strip()


def _cache_key(sql, params):
    normalized = _normalize_sql(sql)
    key = f"{rpd_path}|{normalized}|{params!r}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def query_df(sql, params=None):
    key = _cache_key(sql, params)
    with _query_cache_lock:
        cached = _query_cache.get(key)
        if cached is not None:
            return cached.copy()

    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()

    with _query_cache_lock:
        _query_cache[key] = df.copy()

    return df


def query_df_indexed(sql, params=None):
    """Run a query against a fresh connection with the indexed temp tables
    (tmp_api, tmp_api_ops) available. Each call gets its own connection so
    concurrent requests don't serialize on a single shared connection.
    """
    conn = get_connection()
    try:
        _ensure_indexes_on(conn)
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def table_exists(name):
    if rpd_path is None:
        return False
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type IN ('table','view') AND name=?",
            (name,),
        )
        return cur.fetchone()[0] > 0
    finally:
        conn.close()


def column_exists(table, column):
    if rpd_path is None:
        return False
    conn = get_connection()
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cur.fetchall())
    finally:
        conn.close()


def has_annotations():
    return column_exists("rocpd_api", "domain_id")


def has_torch_ops():
    if not has_annotations():
        return False
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT count(*) FROM rocpd_api JOIN rocpd_string ON rocpd_string.id = rocpd_api.domain_id "
            "WHERE rocpd_string.string = 'torch' LIMIT 1"
        )
        return cur.fetchone()[0] > 0
    finally:
        conn.close()


def set_rpd_path(path):
    global rpd_path, _persistent_conn, _indexes_ready
    if _persistent_conn is not None:
        _persistent_conn.close()
        _persistent_conn = None
    _indexes_ready = False
    rpd_path = path
    with _query_cache_lock:
        _query_cache.clear()
