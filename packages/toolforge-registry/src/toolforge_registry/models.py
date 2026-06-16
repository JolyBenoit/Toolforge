from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


# --- exceptions ---


class RegistryError(Exception):
    pass


class UsecaseExistsError(RegistryError):
    def __init__(self, usecase_id: str) -> None:
        super().__init__(f"Use case already exists: {usecase_id!r}")
        self.usecase_id = usecase_id


class UsecaseNotFoundError(RegistryError):
    def __init__(self, usecase_id: str) -> None:
        super().__init__(f"Use case not found: {usecase_id!r}")
        self.usecase_id = usecase_id


class RunNotFoundError(RegistryError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run not found: {run_id!r}")
        self.run_id = run_id


class ToolNotFoundError(RegistryError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Tool not found: {name!r}")
        self.name = name


class ToolExistsError(RegistryError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Tool already exists: {name!r}. Use update_tool to add a new version.")
        self.name = name


class RunLockedError(RegistryError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id!r} is not in draft status; fork it to iterate or unlock it")
        self.run_id = run_id


class SandboxNotValidatedError(RegistryError):
    def __init__(self, name: str, version: int) -> None:
        super().__init__(
            f"Tool {name!r} v{version} has not passed sandbox validation; validate before promoting"
        )
        self.name = name
        self.version = version


class RunNotValidatedError(RegistryError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id!r} is not validated; only validated or in_production runs can be forked")
        self.run_id = run_id


# --- data classes ---


@dataclass
class UsecaseInfo:
    usecase_id: str
    prompt: str
    created_at: datetime


@dataclass
class ToolVersionInfo:
    version: int
    sandbox_validated: bool
    created_at: datetime
    requirements: list[str] = field(default_factory=list)


@dataclass
class ToolInfo:
    name: str
    description: str
    status: Literal["active", "deprecated"]
    active_version: int | None
    versions: list[ToolVersionInfo]
    created_at: datetime
    updated_at: datetime


@dataclass
class RunInfo:
    run_id: str
    usecase_id: str
    status: Literal["draft", "validated", "in_production"]
    created_at: datetime
    validated_at: datetime | None
    forked_from: str | None
    promoted_to_production_at: datetime | None = None
    tool_count: int = 0
