from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProviderConfig:
    api_key_env: str = ""
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 120.0  # seconds; raise for large-context workloads

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if not self.api_key_env:
            raise EnvironmentError(
                "Provider has neither 'api_key' nor 'api_key_env' configured."
            )
        value = os.environ.get(self.api_key_env)
        if not value:
            raise EnvironmentError(f"Environment variable {self.api_key_env!r} is not set")
        return value


@dataclass
class AgentLLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    system_prompt_file: Path


@dataclass
class LLMToolConfig:
    provider: str
    model: str
    max_tokens: int
    temperature: float


@dataclass
class LLMConfig:
    creator: AgentLLMConfig
    consumer: AgentLLMConfig
    judge: AgentLLMConfig
    providers: dict[str, ProviderConfig]
    tools: dict[str, LLMToolConfig]


@dataclass
class MCPEndpointConfig:
    host: str
    port: int


@dataclass
class MCPConfig:
    creator: MCPEndpointConfig
    usecase: MCPEndpointConfig


@dataclass
class SandboxConfig:
    timeout_seconds: int
    mode: str = "uv"
    image: str = "toolforge-sandbox:latest"
    # uv mode: build one persistent venv per run instead of an isolated env per
    # call. Makes validated/in_production tool calls fast and stable across a run.
    persistent_venv: bool = True


@dataclass
class TelemetryConfig:
    otel_enabled: bool
    otel_endpoint: str
    dsn: str = ""  # Postgres DSN for in_production telemetry (toolforge-telemetry)


@dataclass
class TUIConfig:
    theme: str
    log_level: str


@dataclass
class JudgeConfig:
    # Max number of tasks the static judge evaluates concurrently. The static
    # judge is network-bound (one stateless LLM call per task), so this bounds
    # in-flight LLM requests — raise it to go faster, lower it to stay under an
    # API's rate limit.
    max_concurrency: int = 6


@dataclass
class Config:
    llm: LLMConfig
    mcp: MCPConfig
    sandbox: SandboxConfig
    telemetry: TelemetryConfig
    tui: TUIConfig
    judge: JudgeConfig


def load_config(path: Path) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _parse_config(data, path.parent)


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_llm_tool_configs(config: Config) -> dict[str, dict[str, Any]]:
    """Return llm_tool_configs dict ready to pass to Sandbox.run()."""
    result: dict[str, dict[str, Any]] = {}
    for name, tool_cfg in config.llm.tools.items():
        provider = config.llm.providers.get(tool_cfg.provider)
        if provider is None:
            raise ValueError(
                f"LLM tool {name!r} references unknown provider {tool_cfg.provider!r}"
            )
        result[name] = {
            "provider": tool_cfg.provider,
            "model": tool_cfg.model,
            "max_tokens": tool_cfg.max_tokens,
            "temperature": tool_cfg.temperature,
            "api_key": provider.resolve_api_key(),
            "base_url": provider.base_url,
            "timeout": provider.timeout,
        }
    return result


# --- internal parsers ---


def _parse_agent_config(data: dict[str, Any], config_dir: Path) -> AgentLLMConfig:
    prompt_path = Path(data["system_prompt_file"])
    if not prompt_path.is_absolute():
        prompt_path = config_dir / prompt_path
    return AgentLLMConfig(
        provider=data["provider"],
        model=data["model"],
        temperature=float(data["temperature"]),
        max_tokens=int(data["max_tokens"]),
        system_prompt_file=prompt_path,
    )


def _parse_provider(data: dict[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        api_key_env=data.get("api_key_env", ""),
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
        timeout=float(data.get("timeout", 120.0)),
    )


def _parse_llm_tool(data: dict[str, Any]) -> LLMToolConfig:
    return LLMToolConfig(
        provider=data["provider"],
        model=data["model"],
        max_tokens=int(data.get("max_tokens", 1024)),
        temperature=float(data.get("temperature", 0.3)),
    )


def _parse_config(data: dict[str, Any], config_dir: Path) -> Config:
    llm_data = data["llm"]
    providers = {name: _parse_provider(cfg) for name, cfg in llm_data.get("providers", {}).items()}
    tools = {name: _parse_llm_tool(cfg) for name, cfg in llm_data.get("tools", {}).items()}
    creator = _parse_agent_config(llm_data["creator"], config_dir)
    # The judge runs on the same LLM as the creator by default; a [llm.judge]
    # section overrides any field (provider, model, system prompt, …).
    judge = (
        _parse_agent_config(llm_data["judge"], config_dir)
        if "judge" in llm_data
        else creator
    )
    llm = LLMConfig(
        creator=creator,
        consumer=_parse_agent_config(llm_data["consumer"], config_dir),
        judge=judge,
        providers=providers,
        tools=tools,
    )

    mcp_data = data.get("mcp", {})
    mcp = MCPConfig(
        creator=MCPEndpointConfig(**mcp_data.get("creator", {"host": "localhost", "port": 8765})),
        usecase=MCPEndpointConfig(**mcp_data.get("usecase", {"host": "localhost", "port": 8766})),
    )

    sb = data.get("sandbox", {})
    sandbox = SandboxConfig(
        timeout_seconds=int(sb.get("timeout_seconds", 30)),
        mode=sb.get("mode", "uv"),
        image=sb.get("image", "toolforge-sandbox:latest"),
        persistent_venv=bool(sb.get("persistent_venv", True)),
    )

    tel = data.get("telemetry", {})
    telemetry = TelemetryConfig(
        otel_enabled=bool(tel.get("otel_enabled", False)),
        otel_endpoint=tel.get("otel_endpoint", ""),
        dsn=tel.get("dsn", ""),
    )

    tui_d = data.get("tui", {})
    tui = TUIConfig(
        theme=tui_d.get("theme", "dark"),
        log_level=tui_d.get("log_level", "INFO"),
    )

    judge_d = data.get("judge", {})
    judge_cfg = JudgeConfig(max_concurrency=int(judge_d.get("max_concurrency", 6)))

    return Config(
        llm=llm, mcp=mcp, sandbox=sandbox, telemetry=telemetry, tui=tui,
        judge=judge_cfg,
    )
