from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxResult:
    """Result of executing a tool handler inside the sandbox."""

    output: Any        # deserialized return value of run(args), or None on failure
    stdout: str        # full stdout captured from the container
    stderr: str        # stderr (handler prints + error tracebacks)
    duration_ms: float
    exit_code: int     # 0 = success, 1 = handler error, -1 = timeout/infra error
    nested_llm_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0
