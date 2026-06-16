from pathlib import Path

import pytest

from toolforge_core.config import load_config

MINIMAL_TOML = """
[llm.creator]
provider = "anthropic"
model = "claude-opus-4-7"
temperature = 0.7
max_tokens = 4096
system_prompt_file = "prompts/creator_system.md"

[llm.consumer]
provider = "zai"
model = "qwen-coder-32b"
temperature = 0.5
max_tokens = 8192
system_prompt_file = "prompts/consumer_system.md"

[llm.providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[llm.providers.zai]
api_key_env = "ZAI_API_KEY"
base_url = "https://api.zai.ai/v1"

[llm.tools.default]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
max_tokens = 2048
temperature = 0.3

[mcp.creator]
host = "localhost"
port = 8765

[mcp.usecase]
host = "localhost"
port = 8766

[sandbox]
timeout_seconds = 30

[telemetry]
otel_enabled = false
otel_endpoint = ""

[tui]
theme = "dark"
log_level = "INFO"
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    f = tmp_path / "config.toml"
    f.write_text(MINIMAL_TOML, encoding="utf-8")
    return f


def test_load_creator(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.llm.creator.provider == "anthropic"
    assert cfg.llm.creator.model == "claude-opus-4-7"
    assert cfg.llm.creator.temperature == 0.7
    assert cfg.llm.creator.max_tokens == 4096
    assert cfg.llm.creator.system_prompt_file.name == "creator_system.md"


def test_load_consumer(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.llm.consumer.provider == "zai"
    assert cfg.llm.consumer.max_tokens == 8192


def test_load_providers(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert "anthropic" in cfg.llm.providers
    assert cfg.llm.providers["anthropic"].api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.llm.providers["anthropic"].base_url is None
    assert cfg.llm.providers["zai"].base_url == "https://api.zai.ai/v1"


def test_load_mcp(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.mcp.creator.host == "localhost"
    assert cfg.mcp.creator.port == 8765
    assert cfg.mcp.usecase.port == 8766


def test_load_sandbox(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.sandbox.timeout_seconds == 30


def test_load_llm_tools(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert "default" in cfg.llm.tools
    assert cfg.llm.tools["default"].provider == "anthropic"
    assert cfg.llm.tools["default"].model == "claude-haiku-4-5-20251001"
    assert cfg.llm.tools["default"].max_tokens == 2048


def test_load_no_llm_tools_is_empty_dict(tmp_path: Path) -> None:
    toml = MINIMAL_TOML.replace("[llm.tools.default]\n", "").replace(
        "provider = \"anthropic\"\nmodel = \"claude-haiku-4-5-20251001\"\nmax_tokens = 2048\ntemperature = 0.3\n", ""
    )
    f = tmp_path / "config.toml"
    f.write_text(toml, encoding="utf-8")
    cfg = load_config(f)
    assert cfg.llm.tools == {}


def test_system_prompt_path_is_absolute(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.llm.creator.system_prompt_file.is_absolute()
