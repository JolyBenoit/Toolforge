"""SQLite-backed tool metadata store, one DB per run."""
from __future__ import annotations

import json as _json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from .models import (
    SandboxNotValidatedError,
    ToolExistsError,
    ToolInfo,
    ToolNotFoundError,
    ToolVersionInfo,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tools (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',
    active_version INTEGER,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_versions (
    tool_name          TEXT NOT NULL REFERENCES tools(name),
    version            INTEGER NOT NULL,
    handler_path       TEXT NOT NULL,
    schema_path        TEXT NOT NULL,
    requirements       TEXT NOT NULL DEFAULT '[]',
    sandbox_validated  INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (tool_name, version)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunDB:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._init()

    # --- connection management ---

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            # Migration: add requirements column to existing DBs that predate it
            cols = {row[1] for row in conn.execute("PRAGMA table_info(tool_versions)")}
            if "requirements" not in cols:
                conn.execute(
                    "ALTER TABLE tool_versions ADD COLUMN requirements TEXT NOT NULL DEFAULT '[]'"
                )
                conn.commit()
        finally:
            conn.close()

    # --- version numbering ---

    def next_version(self, name: str) -> int:
        """Return the next version number for a tool (1 if new, N+1 if existing)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT MAX(version) AS m FROM tool_versions WHERE tool_name=?", (name,)
            ).fetchone()
            return (row["m"] or 0) + 1
        finally:
            conn.close()

    # --- write operations ---

    def insert_version(
        self,
        name: str,
        description: str,
        version: int,
        handler_path: str,
        schema_path: str,
        requirements: list[str],
        *,
        require_new: bool = False,
        require_existing: bool = False,
    ) -> ToolVersionInfo:
        """Insert a new version record, creating the tool row if needed."""
        now = _now()
        reqs_json = _json.dumps(requirements)
        with self._tx() as conn:
            existing = conn.execute("SELECT name FROM tools WHERE name=?", (name,)).fetchone()
            if require_new and existing:
                raise ToolExistsError(name)
            if require_existing and not existing:
                raise ToolNotFoundError(name)

            if existing is None:
                conn.execute(
                    "INSERT INTO tools (name, description, status, active_version, created_at, updated_at)"
                    " VALUES (?, ?, 'active', NULL, ?, ?)",
                    (name, description, now, now),
                )
            else:
                conn.execute(
                    "UPDATE tools SET description=?, updated_at=? WHERE name=?",
                    (description, now, name),
                )

            conn.execute(
                "INSERT INTO tool_versions"
                " (tool_name, version, handler_path, schema_path, requirements, sandbox_validated, created_at)"
                " VALUES (?, ?, ?, ?, ?, 0, ?)",
                (name, version, handler_path, schema_path, reqs_json, now),
            )
        return ToolVersionInfo(
            version=version,
            sandbox_validated=False,
            created_at=datetime.fromisoformat(now),
            requirements=requirements,
        )

    def mark_sandbox_validated(self, name: str, version: int) -> None:
        with self._tx() as conn:
            n = conn.execute(
                "UPDATE tool_versions SET sandbox_validated=1"
                " WHERE tool_name=? AND version=?",
                (name, version),
            ).rowcount
            if n == 0:
                raise ToolNotFoundError(f"{name}:v{version}")

    def promote_tool(self, name: str, version: int) -> ToolInfo:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT sandbox_validated FROM tool_versions WHERE tool_name=? AND version=?",
                (name, version),
            ).fetchone()
            if row is None:
                raise ToolNotFoundError(f"{name}:v{version}")
            if not row["sandbox_validated"]:
                raise SandboxNotValidatedError(name, version)
            conn.execute(
                "UPDATE tools SET active_version=?, updated_at=? WHERE name=?",
                (version, _now(), name),
            )
        return self.get_tool(name)

    def deprecate_tool(self, name: str) -> ToolInfo:
        with self._tx() as conn:
            n = conn.execute(
                "UPDATE tools SET status='deprecated', active_version=NULL, updated_at=?"
                " WHERE name=?",
                (_now(), name),
            ).rowcount
            if n == 0:
                raise ToolNotFoundError(name)
        return self.get_tool(name)

    # --- read operations ---

    def get_tool(self, name: str) -> ToolInfo:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM tools WHERE name=?", (name,)).fetchone()
            if row is None:
                raise ToolNotFoundError(name)
            versions = [
                ToolVersionInfo(
                    version=v["version"],
                    sandbox_validated=bool(v["sandbox_validated"]),
                    created_at=datetime.fromisoformat(v["created_at"]),
                    requirements=_json.loads(v["requirements"]),
                )
                for v in conn.execute(
                    "SELECT * FROM tool_versions WHERE tool_name=? ORDER BY version", (name,)
                )
            ]
            return ToolInfo(
                name=row["name"],
                description=row["description"],
                status=row["status"],
                active_version=row["active_version"],
                versions=versions,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
        finally:
            conn.close()

    def list_tools(self) -> list[ToolInfo]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT name FROM tools ORDER BY created_at").fetchall()
        finally:
            conn.close()
        return [self.get_tool(r["name"]) for r in rows]

    def get_handler_path(self, name: str, version: int | None = None) -> str:
        return self._get_path_field(name, version, "handler_path")

    def get_schema_path(self, name: str, version: int | None = None) -> str:
        return self._get_path_field(name, version, "schema_path")

    def get_requirements(self, name: str, version: int | None = None) -> list[str]:
        raw = self._get_path_field(name, version, "requirements")
        return _json.loads(raw)

    def _get_path_field(self, name: str, version: int | None, field: str) -> str:
        conn = self._connect()
        try:
            if version is None:
                row = conn.execute(
                    "SELECT active_version FROM tools WHERE name=?", (name,)
                ).fetchone()
                if row is None:
                    raise ToolNotFoundError(name)
                version = row["active_version"]
                if version is None:
                    raise ToolNotFoundError(f"{name} (no active version promoted yet)")
            row = conn.execute(
                f"SELECT {field} FROM tool_versions WHERE tool_name=? AND version=?",  # noqa: S608
                (name, version),
            ).fetchone()
            if row is None:
                raise ToolNotFoundError(f"{name}:v{version}")
            return str(row[field])
        finally:
            conn.close()
