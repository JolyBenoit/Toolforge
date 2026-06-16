"""Tests for the RUNNER_SCRIPT.  No Docker required — executed with the host Python."""
import json
import subprocess
import sys

from toolforge_sandbox._runner import RUNNER_SCRIPT


def _run(
    handler_source: str,
    args: dict,
    llm_configs: dict | None = None,
) -> tuple[int, dict | None, bytes]:
    payload = json.dumps({
        "handler_source": handler_source,
        "args": args,
        "llm_configs": llm_configs,
    })
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER_SCRIPT],
        input=payload.encode(),
        capture_output=True,
        timeout=10,
    )
    last = proc.stdout.strip().rsplit(b"\n", 1)[-1] if proc.stdout.strip() else b""
    parsed = json.loads(last) if last else None
    return proc.returncode, parsed, proc.stderr


def test_integer_return() -> None:
    rc, out, _ = _run("def run(args):\n    return args['x'] + 1", {"x": 41})
    assert rc == 0
    assert out is not None
    assert out["output"] == 42
    assert out["error"] is None


def test_string_return() -> None:
    rc, out, _ = _run("def run(args):\n    return 'hello'", {})
    assert rc == 0
    assert out["output"] == "hello"


def test_dict_return() -> None:
    rc, out, _ = _run("def run(args):\n    return {'k': [1, 2, 3]}", {})
    assert rc == 0
    assert out["output"] == {"k": [1, 2, 3]}


def test_none_return() -> None:
    rc, out, _ = _run("def run(args):\n    pass", {})
    assert rc == 0
    assert out["output"] is None


def test_handler_print_goes_to_stderr_not_stdout() -> None:
    handler = "def run(args):\n    print('debug line')\n    return 99"
    payload = json.dumps({"handler_source": handler, "args": {}, "llm_configs": None})
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER_SCRIPT],
        input=payload.encode(),
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0
    last = json.loads(proc.stdout.strip().rsplit(b"\n", 1)[-1])
    assert last["output"] == 99
    assert b"debug line" in proc.stderr


def test_runtime_error_exits_1() -> None:
    rc, out, stderr = _run("def run(args):\n    raise ValueError('bad')", {})
    assert rc == 1
    assert out is not None
    assert out["output"] is None
    assert out["error"]["type"] == "ValueError"
    assert "bad" in out["error"]["message"]
    assert "Traceback" in out["error"]["traceback"]


def test_missing_run_function() -> None:
    rc, out, _ = _run("result = 42  # no run() defined", {})
    assert rc == 1
    assert out["error"]["type"] == "AttributeError"


def test_non_serialisable_return_coerced_to_str() -> None:
    handler = "def run(args):\n    class Obj: pass\n    return Obj()"
    rc, out, _ = _run(handler, {})
    assert rc == 0
    assert isinstance(out["output"], str)


def test_args_forwarded_correctly() -> None:
    handler = "def run(args):\n    return args['a'] + args['b']"
    rc, out, _ = _run(handler, {"a": 10, "b": 32})
    assert rc == 0
    assert out["output"] == 42


def test_import_stdlib_allowed() -> None:
    handler = "import math\ndef run(args):\n    return math.factorial(5)"
    rc, out, _ = _run(handler, {})
    assert rc == 0
    assert out["output"] == 120


def test_llm_injected_in_namespace() -> None:
    handler = "def run(args):\n    return llm is not None"
    rc, out, _ = _run(handler, {})
    assert rc == 0
    assert out["output"] is True


def test_llm_registry_is_injected() -> None:
    handler = "def run(args):\n    return type(llm).__name__"
    rc, out, _ = _run(handler, {})
    assert rc == 0
    assert out["output"] == "_LLMRegistry"


def test_llm_unknown_tool_raises_attribute_error() -> None:
    handler = "def run(args):\n    return llm.missing.complete('hello')"
    rc, out, _ = _run(handler, {}, llm_configs=None)
    assert rc == 1
    assert out["error"]["type"] == "AttributeError"
    assert "missing" in out["error"]["message"]


def test_llm_unknown_tool_key_raises_key_error() -> None:
    handler = "def run(args):\n    return llm['missing'].complete('hello')"
    rc, out, _ = _run(handler, {}, llm_configs=None)
    assert rc == 1
    assert out["error"]["type"] == "KeyError"


def test_llm_tool_complete_is_callable() -> None:
    handler = "def run(args):\n    return callable(llm.default.complete)"
    rc, out, _ = _run(handler, {}, llm_configs={"default": {
        "provider": "openai", "model": "x", "api_key": "k", "max_tokens": 10,
    }})
    assert rc == 0
    assert out["output"] is True


def test_llm_tool_chat_is_callable() -> None:
    handler = "def run(args):\n    return callable(llm.default.chat)"
    rc, out, _ = _run(handler, {}, llm_configs={"default": {
        "provider": "openai", "model": "x", "api_key": "k", "max_tokens": 10,
    }})
    assert rc == 0
    assert out["output"] is True


def test_llm_dict_access_same_as_attr() -> None:
    handler = "def run(args):\n    return llm['default'] is llm.default"
    rc, out, _ = _run(handler, {}, llm_configs={"default": {
        "provider": "openai", "model": "x", "api_key": "k", "max_tokens": 10,
    }})
    assert rc == 0
    assert out["output"] is True


def test_llm_error_message_lists_available_tools() -> None:
    handler = "def run(args):\n    return llm.unknown.complete('x')"
    rc, out, _ = _run(handler, {}, llm_configs={"default": {
        "provider": "openai", "model": "x", "api_key": "k", "max_tokens": 10,
    }})
    assert rc == 1
    assert "default" in out["error"]["message"]
