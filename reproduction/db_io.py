from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TableRef:
    schema: str
    name: str

    @classmethod
    def parse(cls, raw: str, default_schema: str) -> "TableRef":
        raw = raw.strip()
        if "." in raw:
            schema, name = raw.split(".", 1)
            return cls(schema=schema.strip('"'), name=name.strip('"'))
        return cls(schema=default_schema, name=raw.strip('"'))

    def label(self) -> str:
        if self.schema in ("", "main"):
            return self.name
        return f"{self.schema}.{self.name}"


class Database:
    def __init__(self, backend: str, conn: Any) -> None:
        self.backend = backend
        self.conn = conn

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.conn.close()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ddmm_to_decimal(value: Any) -> float:
    v = float(value)
    deg = int(v)
    minutes = round((v - deg) * 100)
    return deg + minutes / 60.0


def station_metadata_to_decimal(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("longitude", "latitude"):
        if data.get(key) is not None:
            data[key] = ddmm_to_decimal(data[key])
    return data


def connect(config: dict[str, Any]):
    db_cfg = config.get("database", {})
    backend = (db_cfg.get("backend") or "").strip().lower()
    if backend == "sqlite" or db_cfg.get("sqlite_path"):
        path = Path(db_cfg.get("sqlite_path") or os.environ.get("SQLITE_PATH") or "")
        if not path:
            raise ValueError("Missing SQLite path: database.sqlite_path")
        uri = str(path)
        if db_cfg.get("read_only", True):
            options = ["mode=ro"]
            if db_cfg.get("immutable", False):
                options.append("immutable=1")
            uri = f"file:{path}?{'&'.join(options)}"
        conn = sqlite3.connect(uri, uri=uri.startswith("file:"))
        conn.row_factory = sqlite3.Row
        return Database("sqlite", conn)

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "Missing PostgreSQL driver. Install with: python3 -m pip install -r requirements.txt"
        ) from exc

    url = (db_cfg.get("url") or os.environ.get("DATABASE_URL") or "").strip()
    if url:
        return Database("postgres", psycopg.connect(url, row_factory=dict_row))

    kwargs = {
        "host": db_cfg.get("host") or os.environ.get("PGHOST"),
        "port": int(db_cfg.get("port") or os.environ.get("PGPORT") or 5432),
        "dbname": db_cfg.get("dbname") or os.environ.get("PGDATABASE"),
        "user": db_cfg.get("user") or os.environ.get("PGUSER"),
        "password": db_cfg.get("password") or os.environ.get("PGPASSWORD"),
        "row_factory": dict_row,
    }
    missing = [k for k in ("host", "dbname", "user") if not kwargs.get(k)]
    if missing:
        raise ValueError(
            "Missing PostgreSQL connection settings: "
            + ", ".join(missing)
            + ". Fill config.yaml or set PG* environment variables."
        )
    return Database("postgres", psycopg.connect(**kwargs))


def candidate_tables(db: Database, schema: str, patterns: list[str]) -> list[dict[str, Any]]:
    patterns = [p.lower() for p in patterns if p]
    if db.backend == "sqlite":
        rows = [
            dict(row)
            for row in db.conn.execute(
                """
                SELECT 'main' AS table_schema, name AS table_name
                FROM sqlite_master
                WHERE type IN ('table', 'view')
                ORDER BY name
                """
            ).fetchall()
        ]
    else:
        with db.conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                  AND table_schema = %s
                ORDER BY table_schema, table_name
                """,
                (schema,),
            )
            rows = cur.fetchall()

    candidates = []
    for row in rows:
        table_name = row["table_name"]
        haystack = table_name.lower()
        if not patterns or any(p in haystack for p in patterns):
            candidates.append(row)
    return candidates


def table_columns(db: Database, table: TableRef) -> list[dict[str, Any]]:
    if db.backend == "sqlite":
        pragma = f"PRAGMA table_info({_sqlite_table_expr(table)})"
        rows = db.conn.execute(pragma).fetchall()
        return [
            {
                "ordinal_position": int(row["cid"]) + 1,
                "column_name": row["name"],
                "data_type": row["type"],
                "udt_name": row["type"],
                "is_nullable": "NO" if row["notnull"] else "YES",
                "column_default": row["dflt_value"],
                "primary_key": int(row["pk"]),
            }
            for row in rows
        ]

    with db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                ordinal_position,
                column_name,
                data_type,
                udt_name,
                is_nullable,
                column_default,
                0 AS primary_key
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table.schema, table.name),
        )
        return cur.fetchall()


def table_sample(db: Database, table: TableRef, limit: int = 5) -> list[dict[str, Any]]:
    if db.backend == "sqlite":
        query = f"SELECT * FROM {_sqlite_table_expr(table)} LIMIT ?"
        return [dict(row) for row in db.conn.execute(query, (limit,)).fetchall()]

    from psycopg import sql

    query = sql.SQL("SELECT * FROM {}.{} LIMIT {}").format(
        sql.Identifier(table.schema),
        sql.Identifier(table.name),
        sql.Literal(limit),
    )
    with db.conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def table_count(db: Database, table: TableRef) -> int | None:
    if db.backend == "sqlite":
        row = db.conn.execute(
            f"SELECT COUNT(*) AS n FROM {_sqlite_table_expr(table)}"
        ).fetchone()
        return None if row is None else int(row["n"])

    from psycopg import sql

    query = sql.SQL("SELECT COUNT(*) AS n FROM {}.{}").format(
        sql.Identifier(table.schema),
        sql.Identifier(table.name),
    )
    with db.conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()
    return None if row is None else int(row["n"])


def configured_tables(config: dict[str, Any]) -> list[TableRef]:
    db_cfg = config.get("database", {})
    schema = db_cfg.get("schema") or "public"
    names = [db_cfg.get("history_table") or "", db_cfg.get("forecast_table") or ""]
    return [TableRef.parse(name, schema) for name in names if name.strip()]


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sqlite_table_expr(table: TableRef) -> str:
    if table.schema and table.schema != "main":
        return (
            f"{_quote_sqlite_identifier(table.schema)}."
            f"{_quote_sqlite_identifier(table.name)}"
        )
    return _quote_sqlite_identifier(table.name)
