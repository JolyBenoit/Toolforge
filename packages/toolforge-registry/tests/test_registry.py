from pathlib import Path

import pytest

from toolforge_registry import (
    Registry,
    RegistryError,
    RunLockedError,
    RunNotFoundError,
    RunNotValidatedError,
    SandboxNotValidatedError,
    ToolExistsError,
    ToolNotFoundError,
    UsecaseExistsError,
    UsecaseNotFoundError,
)

_PROMPT = "Process invoices from email attachments and store them in a spreadsheet."
_HANDLER_V1 = "def run(args):\n    return 'v1'"
_HANDLER_V2 = "def run(args):\n    return 'v2'"
_SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}


@pytest.fixture
def reg(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "data")


@pytest.fixture
def uc(reg: Registry) -> str:
    reg.create_usecase("uc_test", _PROMPT)
    return "uc_test"


@pytest.fixture
def run(reg: Registry, uc: str):
    return reg.create_run(uc)


# --- use case ---


def test_create_usecase(reg: Registry) -> None:
    info = reg.create_usecase("uc_1", _PROMPT)
    assert info.usecase_id == "uc_1"
    assert info.prompt == _PROMPT


def test_get_usecase(reg: Registry, uc: str) -> None:
    info = reg.get_usecase(uc)
    assert info.prompt == _PROMPT


def test_list_usecases(reg: Registry) -> None:
    reg.create_usecase("uc_a", "a")
    reg.create_usecase("uc_b", "b")
    ids = [u.usecase_id for u in reg.list_usecases()]
    assert "uc_a" in ids and "uc_b" in ids


def test_duplicate_usecase_raises(reg: Registry, uc: str) -> None:
    with pytest.raises(UsecaseExistsError):
        reg.create_usecase(uc, "duplicate")


def test_missing_usecase_raises(reg: Registry) -> None:
    with pytest.raises(UsecaseNotFoundError):
        reg.get_usecase("no_such")


# --- rename use case ---


def test_rename_usecase_moves_folder_and_restamps(reg: Registry, uc: str) -> None:
    run = reg.create_run(uc)
    info = reg.rename_usecase(uc, "uc_renamed")

    assert info.usecase_id == "uc_renamed"
    assert info.prompt == _PROMPT
    assert not reg.usecase_exists(uc)
    assert reg.usecase_exists("uc_renamed")
    # The run survives the move and its metadata points at the new id.
    moved = reg.get_run("uc_renamed", run.run_id)
    assert moved.usecase_id == "uc_renamed"
    with pytest.raises(UsecaseNotFoundError):
        reg.list_runs(uc)


def test_rename_usecase_drops_per_run_venv(reg: Registry, uc: str) -> None:
    run = reg.create_run(uc)
    venv = reg._run_dir(uc, run.run_id) / ".venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /old/path", encoding="utf-8")

    reg.rename_usecase(uc, "uc_renamed")

    assert not (reg._run_dir("uc_renamed", run.run_id) / ".venv").exists()


def test_rename_usecase_to_existing_raises(reg: Registry, uc: str) -> None:
    reg.create_usecase("uc_other", "x")
    with pytest.raises(UsecaseExistsError):
        reg.rename_usecase(uc, "uc_other")


def test_rename_missing_usecase_raises(reg: Registry) -> None:
    with pytest.raises(UsecaseNotFoundError):
        reg.rename_usecase("no_such", "whatever")


def test_rename_usecase_rejects_invalid_id(reg: Registry, uc: str) -> None:
    with pytest.raises(RegistryError):
        reg.rename_usecase(uc, "bad id!")


def test_rename_usecase_same_id_is_noop(reg: Registry, uc: str) -> None:
    info = reg.rename_usecase(uc, uc)
    assert info.usecase_id == uc
    assert reg.usecase_exists(uc)


# --- runs ---


def test_create_run(reg: Registry, uc: str, run) -> None:
    assert run.status == "draft"
    assert run.usecase_id == uc
    assert run.forked_from is None


def test_list_runs(reg: Registry, uc: str) -> None:
    reg.create_run(uc)
    reg.create_run(uc)
    assert len(reg.list_runs(uc)) == 2


def test_get_run(reg: Registry, uc: str, run) -> None:
    fetched = reg.get_run(uc, run.run_id)
    assert fetched.run_id == run.run_id
    assert fetched.status == "draft"


def test_missing_run_raises(reg: Registry, uc: str) -> None:
    with pytest.raises(RunNotFoundError):
        reg.get_run(uc, "r_no_such")


def test_validate_run(reg: Registry, uc: str, run) -> None:
    validated = reg.validate_run(uc, run.run_id)
    assert validated.status == "validated"
    assert validated.validated_at is not None


def test_validate_already_validated_raises(reg: Registry, uc: str, run) -> None:
    reg.validate_run(uc, run.run_id)
    with pytest.raises(RegistryError):
        reg.validate_run(uc, run.run_id)


def test_fork_run(reg: Registry, uc: str, run) -> None:
    reg.validate_run(uc, run.run_id)
    forked = reg.fork_run(uc, run.run_id)
    assert forked.status == "draft"
    assert forked.forked_from == run.run_id
    assert forked.run_id != run.run_id


def test_fork_unvalidated_run_raises(reg: Registry, uc: str, run) -> None:
    with pytest.raises(RunNotValidatedError):
        reg.fork_run(uc, run.run_id)


def test_promote_run_to_production(reg: Registry, uc: str, run) -> None:
    reg.validate_run(uc, run.run_id)
    prod = reg.promote_run_to_production(uc, run.run_id)
    assert prod.status == "in_production"
    assert prod.promoted_to_production_at is not None


def test_promote_draft_to_production_raises(reg: Registry, uc: str, run) -> None:
    with pytest.raises(RegistryError):
        reg.promote_run_to_production(uc, run.run_id)


def test_validate_already_in_production_raises(reg: Registry, uc: str, run) -> None:
    reg.validate_run(uc, run.run_id)
    reg.promote_run_to_production(uc, run.run_id)
    with pytest.raises(RegistryError):
        reg.validate_run(uc, run.run_id)


def test_unlock_in_production_run(reg: Registry, uc: str, run) -> None:
    reg.validate_run(uc, run.run_id)
    reg.promote_run_to_production(uc, run.run_id)
    unlocked = reg.unlock_run(uc, run.run_id)
    assert unlocked.status == "draft"
    assert unlocked.promoted_to_production_at is None


def test_fork_in_production_run(reg: Registry, uc: str, run) -> None:
    reg.validate_run(uc, run.run_id)
    reg.promote_run_to_production(uc, run.run_id)
    forked = reg.fork_run(uc, run.run_id)
    assert forked.status == "draft"
    assert forked.forked_from == run.run_id


def test_cannot_propose_on_in_production_run(reg: Registry, uc: str, run) -> None:
    reg.validate_run(uc, run.run_id)
    reg.promote_run_to_production(uc, run.run_id)
    with pytest.raises(RunLockedError):
        reg.propose_tool(uc, run.run_id, "new_tool", "desc", _HANDLER_V1, _SCHEMA)


# --- tools: propose / update ---


def test_propose_tool(reg: Registry, uc: str, run) -> None:
    v = reg.propose_tool(uc, run.run_id, "extract", "Extract data", _HANDLER_V1, _SCHEMA)
    assert v.version == 1
    assert not v.sandbox_validated


def test_propose_duplicate_raises(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "Extract data", _HANDLER_V1, _SCHEMA)
    with pytest.raises(ToolExistsError):
        reg.propose_tool(uc, run.run_id, "extract", "Extract data", _HANDLER_V1, _SCHEMA)


def test_update_tool_adds_version(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "v1 desc", _HANDLER_V1, _SCHEMA)
    v2 = reg.update_tool(uc, run.run_id, "extract", "v2 desc", _HANDLER_V2, _SCHEMA)
    assert v2.version == 2


def test_update_nonexistent_raises(reg: Registry, uc: str, run) -> None:
    with pytest.raises(ToolNotFoundError):
        reg.update_tool(uc, run.run_id, "no_tool", "desc", _HANDLER_V1, _SCHEMA)


def test_handler_source_persisted(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "desc", _HANDLER_V1, _SCHEMA)
    reg.mark_sandbox_validated(uc, run.run_id, "extract", 1)
    reg.promote_tool(uc, run.run_id, "extract", 1)
    src = reg.get_handler_source(uc, run.run_id, "extract")
    assert src == _HANDLER_V1


def test_schema_persisted(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "desc", _HANDLER_V1, _SCHEMA)
    reg.mark_sandbox_validated(uc, run.run_id, "extract", 1)
    reg.promote_tool(uc, run.run_id, "extract", 1)
    schema = reg.get_tool_schema(uc, run.run_id, "extract")
    assert schema == _SCHEMA


# --- tools: promote / validate ---


def test_promote_without_sandbox_validation_raises(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "desc", _HANDLER_V1, _SCHEMA)
    with pytest.raises(SandboxNotValidatedError):
        reg.promote_tool(uc, run.run_id, "extract", 1)


def test_promote_after_validation(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "desc", _HANDLER_V1, _SCHEMA)
    reg.mark_sandbox_validated(uc, run.run_id, "extract", 1)
    info = reg.promote_tool(uc, run.run_id, "extract", 1)
    assert info.active_version == 1


def test_deprecate_tool(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "desc", _HANDLER_V1, _SCHEMA)
    info = reg.deprecate_tool(uc, run.run_id, "extract")
    assert info.status == "deprecated"
    assert info.active_version is None


def test_get_active_tools_filters_correctly(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "promoted", "desc", _HANDLER_V1, _SCHEMA)
    reg.mark_sandbox_validated(uc, run.run_id, "promoted", 1)
    reg.promote_tool(uc, run.run_id, "promoted", 1)

    reg.propose_tool(uc, run.run_id, "unpromoted", "desc", _HANDLER_V1, _SCHEMA)

    reg.propose_tool(uc, run.run_id, "deprecated", "desc", _HANDLER_V1, _SCHEMA)
    reg.deprecate_tool(uc, run.run_id, "deprecated")

    active = reg.get_active_tools(uc, run.run_id)
    assert [t.name for t in active] == ["promoted"]


# --- isolation: validated run is immutable ---


def test_cannot_propose_on_validated_run(reg: Registry, uc: str, run) -> None:
    reg.validate_run(uc, run.run_id)
    with pytest.raises(RunLockedError):
        reg.propose_tool(uc, run.run_id, "new_tool", "desc", _HANDLER_V1, _SCHEMA)


def test_cannot_deprecate_on_validated_run(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "desc", _HANDLER_V1, _SCHEMA)
    reg.validate_run(uc, run.run_id)
    with pytest.raises(RunLockedError):
        reg.deprecate_tool(uc, run.run_id, "extract")


# --- fork preserves tool state ---


def test_fork_copies_tools_and_source(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "desc", _HANDLER_V1, _SCHEMA)
    reg.mark_sandbox_validated(uc, run.run_id, "extract", 1)
    reg.promote_tool(uc, run.run_id, "extract", 1)
    reg.validate_run(uc, run.run_id)

    forked = reg.fork_run(uc, run.run_id)
    tools = reg.list_tools(uc, forked.run_id)
    assert len(tools) == 1
    assert tools[0].name == "extract"
    assert tools[0].active_version == 1
    src = reg.get_handler_source(uc, forked.run_id, "extract")
    assert src == _HANDLER_V1


def test_fork_is_independent_from_source(reg: Registry, uc: str, run) -> None:
    reg.propose_tool(uc, run.run_id, "extract", "desc", _HANDLER_V1, _SCHEMA)
    reg.validate_run(uc, run.run_id)

    forked = reg.fork_run(uc, run.run_id)
    # Can modify the fork without affecting the original
    reg.update_tool(uc, forked.run_id, "extract", "v2 desc", _HANDLER_V2, _SCHEMA)
    assert len(reg.get_tool(uc, forked.run_id, "extract").versions) == 2
    assert len(reg.get_tool(uc, run.run_id, "extract").versions) == 1


# --- run-level merged requirements ---


def _promote(reg: Registry, uc: str, run_id: str, name: str, reqs: list[str]) -> None:
    """Propose, validate, and promote a tool with the given requirements."""
    reg.propose_tool(uc, run_id, name, f"{name} desc", _HANDLER_V1, _SCHEMA, reqs)
    reg.mark_sandbox_validated(uc, run_id, name, 1)
    reg.promote_tool(uc, run_id, name, 1)


def test_recompute_writes_requirements_txt(reg: Registry, uc: str, run) -> None:
    _promote(reg, uc, run.run_id, "tool_a", ["pypdf>=3.0", "httpx"])
    from pathlib import Path
    req_file = Path(reg._run_dir(uc, run.run_id)) / "requirements.txt"
    assert req_file.exists()
    content = req_file.read_text(encoding="utf-8")
    assert "pypdf>=3.0" in content
    assert "httpx" in content


def test_get_run_requirements_returns_merged(reg: Registry, uc: str, run) -> None:
    _promote(reg, uc, run.run_id, "tool_a", ["pypdf>=3.0"])
    _promote(reg, uc, run.run_id, "tool_b", ["requests==2.31.0"])
    reqs = reg.get_run_requirements(uc, run.run_id)
    assert "pypdf>=3.0" in reqs
    assert "requests==2.31.0" in reqs


def test_get_run_requirements_deduplicates(reg: Registry, uc: str, run) -> None:
    _promote(reg, uc, run.run_id, "tool_a", ["httpx>=0.27"])
    _promote(reg, uc, run.run_id, "tool_b", ["httpx>=0.27"])
    reqs = reg.get_run_requirements(uc, run.run_id)
    assert reqs.count("httpx>=0.27") == 1


def test_get_run_requirements_empty_when_no_active_tools(reg: Registry, uc: str, run) -> None:
    reqs = reg.get_run_requirements(uc, run.run_id)
    assert reqs == []


def test_recompute_returns_conflicts_for_diverging_specifiers(
    reg: Registry, uc: str, run
) -> None:
    _promote(reg, uc, run.run_id, "tool_a", ["pypdf>=3.0"])
    _promote(reg, uc, run.run_id, "tool_b", ["pypdf>=2.0"])
    _, conflicts = reg.recompute_run_requirements(uc, run.run_id)
    assert len(conflicts) == 1
    assert "pypdf" in conflicts[0]


def test_recompute_no_conflicts_for_identical_specifiers(
    reg: Registry, uc: str, run
) -> None:
    _promote(reg, uc, run.run_id, "tool_a", ["pypdf>=3.0"])
    _promote(reg, uc, run.run_id, "tool_b", ["pypdf>=3.0"])
    _, conflicts = reg.recompute_run_requirements(uc, run.run_id)
    assert conflicts == []


def test_check_requirements_conflicts_detects_version_mismatch(
    reg: Registry, uc: str, run
) -> None:
    _promote(reg, uc, run.run_id, "existing", ["pypdf>=3.0"])
    conflicts = reg.check_requirements_conflicts(
        uc, run.run_id, ["pypdf>=2.0"], tool_name="new_tool"
    )
    assert len(conflicts) == 1
    assert "pypdf" in conflicts[0]


def test_check_requirements_conflicts_no_conflict_same_spec(
    reg: Registry, uc: str, run
) -> None:
    _promote(reg, uc, run.run_id, "existing", ["pypdf>=3.0"])
    conflicts = reg.check_requirements_conflicts(
        uc, run.run_id, ["pypdf>=3.0"], tool_name="new_tool"
    )
    assert conflicts == []


def test_check_requirements_conflicts_ignores_self(
    reg: Registry, uc: str, run
) -> None:
    _promote(reg, uc, run.run_id, "tool_a", ["pypdf>=3.0"])
    # Updating tool_a with a different specifier should not conflict with itself
    conflicts = reg.check_requirements_conflicts(
        uc, run.run_id, ["pypdf>=2.0"], tool_name="tool_a"
    )
    assert conflicts == []


def test_deprecate_removes_from_run_requirements(reg: Registry, uc: str, run) -> None:
    _promote(reg, uc, run.run_id, "tool_a", ["pypdf>=3.0"])
    _promote(reg, uc, run.run_id, "tool_b", ["requests"])
    reg.deprecate_tool(uc, run.run_id, "tool_a")
    reqs = reg.get_run_requirements(uc, run.run_id)
    assert "pypdf>=3.0" not in reqs
    assert "requests" in reqs


def test_fork_copies_run_requirements(reg: Registry, uc: str, run) -> None:
    _promote(reg, uc, run.run_id, "tool_a", ["pypdf>=3.0"])
    reg.validate_run(uc, run.run_id)
    forked = reg.fork_run(uc, run.run_id)
    reqs = reg.get_run_requirements(uc, forked.run_id)
    assert "pypdf>=3.0" in reqs


def test_get_run_requirements_lazy_recompute(reg: Registry, uc: str, run) -> None:
    """requirements.txt absent → get_run_requirements recomputes from DB."""
    _promote(reg, uc, run.run_id, "tool_a", ["httpx"])
    req_file = reg._run_dir(uc, run.run_id) / "requirements.txt"
    req_file.unlink()  # simulate absent file (old run)
    reqs = reg.get_run_requirements(uc, run.run_id)
    assert "httpx" in reqs
