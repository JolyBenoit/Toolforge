from __future__ import annotations

import json
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _pkg_name_from_req(req: str) -> str:
    """Extract normalized package name from a PEP 508 requirement string.

    Examples: 'pypdf>=3.0' -> 'pypdf', 'Pillow[all]>=9' -> 'pillow'.
    Normalizes per PEP 503: lowercase, collapse [-_.] runs to '-'.
    """
    m = re.match(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)", req.strip())
    name = m.group(1) if m else req.strip()
    return re.sub(r"[-_.]+", "-", name).lower()

from .models import (
    RunInfo,
    RunLockedError,
    RunNotFoundError,
    RunNotValidatedError,
    ToolInfo,
    ToolVersionInfo,
    UsecaseExistsError,
    UsecaseInfo,
    UsecaseNotFoundError,
)
from ._db import RunDB


def _new_run_id() -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = secrets.token_hex(3)
    return f"r_{date}_{suffix}"


class Registry:
    """Persistent store of use cases, runs, and tools."""

    def __init__(self, data_root: Path) -> None:
        self._root = data_root
        (data_root / "usecases").mkdir(parents=True, exist_ok=True)

    # --- path helpers ---

    def _uc_dir(self, usecase_id: str) -> Path:
        return self._root / "usecases" / usecase_id

    def _run_dir(self, usecase_id: str, run_id: str) -> Path:
        return self._uc_dir(usecase_id) / "runs" / run_id

    def _db(self, usecase_id: str, run_id: str) -> RunDB:
        return RunDB(self._run_dir(usecase_id, run_id) / "registry.db")

    # --- use case operations ---

    def create_usecase(self, usecase_id: str, prompt: str) -> UsecaseInfo:
        uc_dir = self._uc_dir(usecase_id)
        if uc_dir.exists():
            raise UsecaseExistsError(usecase_id)
        uc_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc)
        (uc_dir / "usecase.json").write_text(
            json.dumps({"usecase_id": usecase_id, "created_at": now.isoformat()}, indent=2),
            encoding="utf-8",
        )
        (uc_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        return UsecaseInfo(usecase_id=usecase_id, prompt=prompt, created_at=now)

    def get_usecase(self, usecase_id: str) -> UsecaseInfo:
        uc_dir = self._uc_dir(usecase_id)
        if not uc_dir.exists():
            raise UsecaseNotFoundError(usecase_id)
        meta = json.loads((uc_dir / "usecase.json").read_text(encoding="utf-8"))
        prompt = (uc_dir / "prompt.md").read_text(encoding="utf-8")
        return UsecaseInfo(
            usecase_id=usecase_id,
            prompt=prompt,
            created_at=datetime.fromisoformat(meta["created_at"]),
        )

    def list_usecases(self) -> list[UsecaseInfo]:
        uc_root = self._root / "usecases"
        result: list[UsecaseInfo] = []
        for p in sorted(uc_root.iterdir()):
            if p.is_dir() and (p / "usecase.json").exists():
                result.append(self.get_usecase(p.name))
        return result

    # --- run operations ---

    def create_run(self, usecase_id: str) -> RunInfo:
        if not self._uc_dir(usecase_id).exists():
            raise UsecaseNotFoundError(usecase_id)
        run_id = _new_run_id()
        run_dir = self._run_dir(usecase_id, run_id)
        run_dir.mkdir(parents=True)
        (run_dir / "tools").mkdir()
        now = datetime.now(timezone.utc)
        _write_run_meta(run_dir, {
            "run_id": run_id,
            "usecase_id": usecase_id,
            "status": "draft",
            "created_at": now.isoformat(),
            "validated_at": None,
            "promoted_to_production_at": None,
            "forked_from": None,
        })
        RunDB(run_dir / "registry.db")  # initialise schema
        return RunInfo(
            run_id=run_id,
            usecase_id=usecase_id,
            status="draft",
            created_at=now,
            validated_at=None,
            promoted_to_production_at=None,
            forked_from=None,
        )

    def get_run(self, usecase_id: str, run_id: str) -> RunInfo:
        run_dir = self._run_dir(usecase_id, run_id)
        if not run_dir.exists():
            raise RunNotFoundError(run_id)
        meta = _read_run_meta(run_dir)
        tool_count = len(self._db(usecase_id, run_id).list_tools())
        return RunInfo(
            run_id=run_id,
            usecase_id=usecase_id,
            status=meta["status"],
            created_at=datetime.fromisoformat(meta["created_at"]),
            validated_at=datetime.fromisoformat(meta["validated_at"]) if meta["validated_at"] else None,
            forked_from=meta.get("forked_from"),
            promoted_to_production_at=(
                datetime.fromisoformat(meta["promoted_to_production_at"])
                if meta.get("promoted_to_production_at")
                else None
            ),
            tool_count=tool_count,
        )

    def list_runs(self, usecase_id: str) -> list[RunInfo]:
        uc_dir = self._uc_dir(usecase_id)
        if not uc_dir.exists():
            raise UsecaseNotFoundError(usecase_id)
        runs_dir = uc_dir / "runs"
        if not runs_dir.exists():
            return []
        return [
            self.get_run(usecase_id, p.name)
            for p in sorted(runs_dir.iterdir())
            if p.is_dir() and (p / "run.json").exists()
        ]

    def unlock_run(self, usecase_id: str, run_id: str) -> RunInfo:
        """Revert a validated or in_production run back to draft, re-enabling tool edits."""
        run_dir = self._run_dir(usecase_id, run_id)
        if not run_dir.exists():
            raise RunNotFoundError(run_id)
        meta = _read_run_meta(run_dir)
        if meta["status"] not in ("validated", "in_production"):
            from .models import RegistryError
            raise RegistryError(f"Run {run_id!r} is already in draft; nothing to unlock")
        meta["status"] = "draft"
        meta["validated_at"] = None
        meta["promoted_to_production_at"] = None
        _write_run_meta(run_dir, meta)
        return self.get_run(usecase_id, run_id)

    def validate_run(self, usecase_id: str, run_id: str) -> RunInfo:
        run_dir = self._run_dir(usecase_id, run_id)
        if not run_dir.exists():
            raise RunNotFoundError(run_id)
        meta = _read_run_meta(run_dir)
        if meta["status"] != "draft":
            from .models import RegistryError
            raise RegistryError(f"Run {run_id!r} is already validated or in production")
        now = datetime.now(timezone.utc)
        meta["status"] = "validated"
        meta["validated_at"] = now.isoformat()
        _write_run_meta(run_dir, meta)
        return self.get_run(usecase_id, run_id)

    def promote_run_to_production(self, usecase_id: str, run_id: str) -> RunInfo:
        """Promote a validated run to in_production status."""
        run_dir = self._run_dir(usecase_id, run_id)
        if not run_dir.exists():
            raise RunNotFoundError(run_id)
        meta = _read_run_meta(run_dir)
        if meta["status"] != "validated":
            from .models import RegistryError
            raise RegistryError(
                f"Run {run_id!r} must be validated before promoting to production (current: {meta['status']!r})"
            )
        now = datetime.now(timezone.utc)
        meta["status"] = "in_production"
        meta["promoted_to_production_at"] = now.isoformat()
        _write_run_meta(run_dir, meta)
        return self.get_run(usecase_id, run_id)

    def fork_run(self, usecase_id: str, from_run_id: str) -> RunInfo:
        from_dir = self._run_dir(usecase_id, from_run_id)
        if not from_dir.exists():
            raise RunNotFoundError(from_run_id)
        from_meta = _read_run_meta(from_dir)
        if from_meta["status"] not in ("validated", "in_production"):
            raise RunNotValidatedError(from_run_id)

        run_id = _new_run_id()
        new_dir = self._run_dir(usecase_id, run_id)
        new_dir.mkdir(parents=True)

        # Copy tool source files and DB (preserving sandbox_validated state)
        shutil.copytree(from_dir / "tools", new_dir / "tools")
        shutil.copy2(from_dir / "registry.db", new_dir / "registry.db")

        now = datetime.now(timezone.utc)
        _write_run_meta(new_dir, {
            "run_id": run_id,
            "usecase_id": usecase_id,
            "status": "draft",
            "created_at": now.isoformat(),
            "validated_at": None,
            "promoted_to_production_at": None,
            "forked_from": from_run_id,
        })
        # Recompute from the copied registry.db so requirements.txt is present.
        self.recompute_run_requirements(usecase_id, run_id)
        return RunInfo(
            run_id=run_id,
            usecase_id=usecase_id,
            status="draft",
            created_at=now,
            validated_at=None,
            forked_from=from_run_id,
        )

    # --- inputs operations ---

    def inputs_dir(self, usecase_id: str) -> Path:
        """Return the inputs directory path for a use case (does not create it)."""
        return self._uc_dir(usecase_id) / "inputs"

    def get_consumer_prompt(self, usecase_id: str) -> str | None:
        """Return the use-case-specific consumer instructions, or None if not set."""
        path = self._uc_dir(usecase_id) / "consumer_prompt.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def set_consumer_prompt(self, usecase_id: str, instructions: str) -> None:
        """Write (or replace) the use-case-specific consumer instructions."""
        if not self._uc_dir(usecase_id).exists():
            raise UsecaseNotFoundError(usecase_id)
        (self._uc_dir(usecase_id) / "consumer_prompt.md").write_text(
            instructions, encoding="utf-8"
        )

    def outputs_dir(self, usecase_id: str) -> Path:
        """Return the outputs directory path for a use case (does not create it)."""
        return self._uc_dir(usecase_id) / "outputs"

    def list_outputs(self, usecase_id: str) -> list[Path]:
        """Return all files in the use case outputs/ folder, sorted by name."""
        d = self.outputs_dir(usecase_id)
        if not d.exists():
            return []
        return sorted(f for f in d.iterdir() if f.is_file())

    def add_input(self, usecase_id: str, src: Path) -> Path:
        """Copy src into the use case inputs/ folder, creating it if needed."""
        if not self._uc_dir(usecase_id).exists():
            raise UsecaseNotFoundError(usecase_id)
        dest_dir = self.inputs_dir(usecase_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        return dest

    def list_inputs(self, usecase_id: str) -> list[Path]:
        """Return all files in the use case inputs/ folder, sorted by name."""
        d = self.inputs_dir(usecase_id)
        if not d.exists():
            return []
        return sorted(f for f in d.iterdir() if f.is_file())

    # --- tool operations ---

    def propose_tool(
        self,
        usecase_id: str,
        run_id: str,
        name: str,
        description: str,
        handler_source: str,
        input_schema: dict[str, Any],
        requirements: list[str] | None = None,
    ) -> ToolVersionInfo:
        """Create a new tool (version 1). Raises ToolExistsError if the tool already exists."""
        self._assert_draft(usecase_id, run_id)
        return self._write_tool_version(
            usecase_id, run_id, name, description, handler_source, input_schema,
            requirements=requirements or [],
            require_new=True,
        )

    def update_tool(
        self,
        usecase_id: str,
        run_id: str,
        name: str,
        description: str,
        handler_source: str,
        input_schema: dict[str, Any],
        requirements: list[str] | None = None,
    ) -> ToolVersionInfo:
        """Add a new version to an existing tool. Raises ToolNotFoundError if absent."""
        self._assert_draft(usecase_id, run_id)
        return self._write_tool_version(
            usecase_id, run_id, name, description, handler_source, input_schema,
            requirements=requirements or [],
            require_existing=True,
        )

    def get_tool(self, usecase_id: str, run_id: str, name: str) -> ToolInfo:
        self._assert_run_exists(usecase_id, run_id)
        return self._db(usecase_id, run_id).get_tool(name)

    def list_tools(self, usecase_id: str, run_id: str) -> list[ToolInfo]:
        self._assert_run_exists(usecase_id, run_id)
        return self._db(usecase_id, run_id).list_tools()

    def get_active_tools(self, usecase_id: str, run_id: str) -> list[ToolInfo]:
        """Non-deprecated tools with a promoted active version — served by the Usecase MCP server."""
        return [
            t for t in self.list_tools(usecase_id, run_id)
            if t.status == "active" and t.active_version is not None
        ]

    def mark_sandbox_validated(
        self, usecase_id: str, run_id: str, name: str, version: int
    ) -> None:
        self._assert_run_exists(usecase_id, run_id)
        self._db(usecase_id, run_id).mark_sandbox_validated(name, version)

    def promote_tool(
        self, usecase_id: str, run_id: str, name: str, version: int
    ) -> ToolInfo:
        self._assert_draft(usecase_id, run_id)
        info = self._db(usecase_id, run_id).promote_tool(name, version)
        self.recompute_run_requirements(usecase_id, run_id)
        return info

    def deprecate_tool(self, usecase_id: str, run_id: str, name: str) -> ToolInfo:
        self._assert_draft(usecase_id, run_id)
        info = self._db(usecase_id, run_id).deprecate_tool(name)
        self.recompute_run_requirements(usecase_id, run_id)
        return info

    def get_tool_requirements(
        self, usecase_id: str, run_id: str, name: str, version: int | None = None
    ) -> list[str]:
        self._assert_run_exists(usecase_id, run_id)
        return self._db(usecase_id, run_id).get_requirements(name, version)

    def get_handler_source(
        self, usecase_id: str, run_id: str, name: str, version: int | None = None
    ) -> str:
        self._assert_run_exists(usecase_id, run_id)
        run_dir = self._run_dir(usecase_id, run_id)
        rel = self._db(usecase_id, run_id).get_handler_path(name, version)
        return (run_dir / rel).read_text(encoding="utf-8")

    def get_tool_schema(
        self, usecase_id: str, run_id: str, name: str, version: int | None = None
    ) -> dict[str, Any]:
        self._assert_run_exists(usecase_id, run_id)
        run_dir = self._run_dir(usecase_id, run_id)
        rel = self._db(usecase_id, run_id).get_schema_path(name, version)
        return json.loads((run_dir / rel).read_text(encoding="utf-8"))

    # --- run-level merged requirements ---

    def get_run_requirements(self, usecase_id: str, run_id: str) -> list[str]:
        """Return the run-level merged requirements.

        Reads requirements.txt; lazily recomputes it if the file is absent
        (backward-compatible with runs created before this feature).
        """
        req_file = self._run_dir(usecase_id, run_id) / "requirements.txt"
        if not req_file.exists():
            merged, _ = self.recompute_run_requirements(usecase_id, run_id)
            return merged
        return [
            line.strip()
            for line in req_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def recompute_run_requirements(
        self, usecase_id: str, run_id: str
    ) -> tuple[list[str], list[str]]:
        """Recompute the run-level merged requirements from all active tools.

        Collects requirements from every promoted tool version, deduplicates
        identical specifiers, and flags packages that appear with *different*
        specifiers across tools as potential conflicts.

        Returns (merged_list, conflict_messages) and writes requirements.txt.
        """
        active = self.get_active_tools(usecase_id, run_id)

        # pkg_name -> ordered dict of {req_str: [tool_name, …]}
        pkg_map: dict[str, dict[str, list[str]]] = {}
        for tool in active:
            for req in self.get_tool_requirements(usecase_id, run_id, tool.name):
                pkg = _pkg_name_from_req(req)
                pkg_map.setdefault(pkg, {}).setdefault(req, []).append(tool.name)

        merged: list[str] = []
        conflicts: list[str] = []

        for pkg, spec_to_tools in pkg_map.items():
            if len(spec_to_tools) > 1:
                detail = "; ".join(
                    f"'{spec}' by {', '.join(repr(t) for t in tools)}"
                    for spec, tools in spec_to_tools.items()
                )
                conflicts.append(f"{pkg}: {detail}")
            # Include every unique specifier — let uv resolve or surface errors.
            merged.extend(spec_to_tools.keys())

        run_dir = self._run_dir(usecase_id, run_id)
        req_file = run_dir / "requirements.txt"
        req_file.write_text(
            "\n".join(merged) + ("\n" if merged else ""),
            encoding="utf-8",
        )
        return merged, conflicts

    def check_requirements_conflicts(
        self,
        usecase_id: str,
        run_id: str,
        new_reqs: list[str],
        tool_name: str,
    ) -> list[str]:
        """Check new_reqs against currently active tools (excluding tool_name).

        Returns a list of human-readable conflict messages (empty = clean).
        Does not modify any state.
        """
        active = self.get_active_tools(usecase_id, run_id)

        existing: dict[str, dict[str, str]] = {}  # pkg -> {spec: tool_name}
        for tool in active:
            if tool.name == tool_name:
                continue
            for req in self.get_tool_requirements(usecase_id, run_id, tool.name):
                pkg = _pkg_name_from_req(req)
                existing.setdefault(pkg, {})[req] = tool.name

        conflicts: list[str] = []
        for req in new_reqs:
            pkg = _pkg_name_from_req(req)
            if pkg not in existing:
                continue
            other_specs = existing[pkg]
            if req not in other_specs:
                detail = ", ".join(
                    f"'{r}' (tool '{t}')" for r, t in other_specs.items()
                )
                conflicts.append(f"{pkg}: this tool needs '{req}' but {detail}")

        return conflicts

    # --- internal helpers ---

    def _assert_run_exists(self, usecase_id: str, run_id: str) -> None:
        if not self._run_dir(usecase_id, run_id).exists():
            raise RunNotFoundError(run_id)

    def _assert_draft(self, usecase_id: str, run_id: str) -> None:
        self._assert_run_exists(usecase_id, run_id)
        meta = _read_run_meta(self._run_dir(usecase_id, run_id))
        if meta["status"] != "draft":
            raise RunLockedError(run_id)

    def _write_tool_version(
        self,
        usecase_id: str,
        run_id: str,
        name: str,
        description: str,
        handler_source: str,
        input_schema: dict[str, Any],
        requirements: list[str],
        require_new: bool = False,
        require_existing: bool = False,
    ) -> ToolVersionInfo:
        run_dir = self._run_dir(usecase_id, run_id)
        db = self._db(usecase_id, run_id)

        version = db.next_version(name)
        tool_dir = run_dir / "tools" / name
        tool_dir.mkdir(parents=True, exist_ok=True)

        handler_rel = f"tools/{name}/v{version}.py"
        schema_rel = f"tools/{name}/v{version}.json"
        (run_dir / handler_rel).write_text(handler_source, encoding="utf-8")
        (run_dir / schema_rel).write_text(json.dumps(input_schema, indent=2), encoding="utf-8")

        return db.insert_version(
            name=name,
            description=description,
            version=version,
            handler_path=handler_rel,
            schema_path=schema_rel,
            requirements=requirements,
            require_new=require_new,
            require_existing=require_existing,
        )


# --- filesystem helpers ---


def _read_run_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _write_run_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
