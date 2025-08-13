import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, Tuple, Union

# Try to load environment variables from a .env file if python-dotenv is installed.
# Fallback to a tiny parser if not installed so local dev still works.

def _load_env_file_if_present(env_path: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path)
        return
    except Exception:
        pass

    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Swallow errors; env loading is best-effort.
        pass

# Resolve project root as the directory containing this file.
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_load_env_file_if_present(_ENV_PATH)

# Database configuration from environment
_DB_HOST = os.getenv("DB_HOST", "localhost")
_DB_PORT = int(os.getenv("DB_PORT", "5432"))
_DB_NAME = os.getenv("DB_NAME", "postgres")
_DB_USER = os.getenv("DB_USER", "postgres")
_DB_PASSWORD = os.getenv("DB_PASSWORD", "")
_DB_SSLMODE = os.getenv("DB_SSLMODE", "prefer")

# Lazy import to avoid hard dependency errors at import time if not installed yet
_psycopg2 = None
_pool = None


def _import_psycopg2():
    global _psycopg2
    if _psycopg2 is None:
        import psycopg2  # type: ignore
        from psycopg2 import pool as pg_pool  # type: ignore
        from psycopg2.extras import DictCursor  # type: ignore
        _psycopg2 = {
            "psycopg2": psycopg2,
            "pool": pg_pool,
            "DictCursor": DictCursor,
        }
    return _psycopg2


def get_dsn() -> str:
    """Build a DSN string from environment variables."""
    return (
        f"dbname={_DB_NAME} user={_DB_USER} password={_DB_PASSWORD} "
        f"host={_DB_HOST} port={_DB_PORT} sslmode={_DB_SSLMODE}"
    )


def init_connection_pool(minconn: int = 1, maxconn: int = 5) -> None:
    """Initialize a global connection pool.

    Safe to call multiple times; subsequent calls are ignored once initialized.
    """
    global _pool
    if _pool is not None:
        return
    m = _import_psycopg2()
    _pool = m["pool"].SimpleConnectionPool(minconn, maxconn, dsn=get_dsn())


def close_connection_pool() -> None:
    """Close the global connection pool if initialized."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def get_connection():
    """Get a database connection, using the pool if available."""
    m = _import_psycopg2()
    if _pool is None:
        return m["psycopg2"].connect(dsn=get_dsn())
    return _pool.getconn()


def put_connection(conn) -> None:
    """Return a connection to the pool or close it if pooling is not active."""
    if _pool is None:
        conn.close()
    else:
        _pool.putconn(conn)


@contextmanager
def get_cursor(dict_rows: bool = True):
    """Context manager yielding a cursor and handling commit/rollback.

    Example:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            print(cur.fetchone())
    """
    m = _import_psycopg2()
    conn = get_connection()
    cursor_factory = m["DictCursor"] if dict_rows else None
    cur = conn.cursor(cursor_factory=cursor_factory)
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.close()
        finally:
            put_connection(conn)


def execute_sql(
    query: str,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    *,
    fetch: Optional[str] = None,
    dict_rows: bool = True,
):
    """Execute a SQL statement with optional fetch.

    - fetch=None: no fetch, returns None
    - fetch="one": fetchone()
    - fetch="all": fetchall()
    """
    with get_cursor(dict_rows=dict_rows) as cur:
        cur.execute(query, params)
        if fetch is None:
            return None
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        raise ValueError("fetch must be None, 'one', or 'all'")


def ensure_pgvector_extension() -> None:
    """Ensure the pgvector extension is available in the current database."""
    execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")


if __name__ == "__main__":
    # Optional smoke test
    init_connection_pool()
    row = execute_sql("SELECT version() AS version", fetch="one")
    print({"connected": True, "version": row["version"] if row else None})
    ensure_pgvector_extension()
    print({"pgvector_ready": True}) 