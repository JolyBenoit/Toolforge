"""Sandbox runner — executed as a subprocess inside the sandbox (uv or Docker).

Protocol:
  stdin  — JSON: {"handler_source": str, "args": dict, "llm_configs": dict | null}
  stdout — exactly one JSON line: {"output": <value>, "error": null | {...}, "nested_llm_calls": [...]}
  stderr — any print() calls from the handler, plus tracebacks

The handler must define a top-level ``run(args)`` function.  Its return value
must be JSON-serialisable; non-serialisable values are coerced to str.

``llm`` is injected into the handler namespace as an ``_LLMRegistry`` instance.
Access named tools via attribute (``llm.default``) or key (``llm["default"]``).
Each tool exposes ``.complete(prompt, **kwargs)`` and ``.chat(messages, **kwargs)``.
No SDK dependency required — all HTTP calls use stdlib urllib.

Nested LLM calls made by tools are captured in ``_nlc_log`` and returned in
the stdout JSON so the production telemetry layer can record them.
"""
import io
import json
import socket
import sys
import time as _time
import traceback
import types
import urllib.error
import urllib.request
from datetime import datetime, timezone

# Fallback per-call LLM timeout (seconds) when the sandbox passes no budget.
_DEFAULT_LLM_TIMEOUT = 60.0


def _http_json(req, model, timeout):
    """Send an HTTP request, returning parsed JSON, with clear failure modes.

    A bare urllib call has no timeout: a slow/hung LLM endpoint would block
    until the *sandbox* kills the whole process (exit_code -1), discarding the
    result and the nested-call log. Bounding each call here turns that into a
    normal handler error with a readable message instead.
    """
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"LLM API error {e.code} from '{model}': {detail}") from None
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        reason = getattr(e, "reason", e)
        raise TimeoutError(
            f"LLM call to '{model}' did not complete within {timeout:.0f}s ({reason})"
        ) from None

_out = sys.stdout
sys.stdout = io.StringIO()

# Module-level log of LLM calls made by tools during this execution.
_nlc_log: list[dict] = []
_nlc_seq: int = 0


class _LLMTool:
    def __init__(self, name, cfg, default_timeout=None):
        self._name = name
        self._cfg = cfg
        self._default_timeout = default_timeout

    def _timeout(self):
        """Per-call timeout: the provider's configured value, capped by the
        sandbox budget so a call can never outlast the process that hosts it."""
        budget = self._default_timeout or _DEFAULT_LLM_TIMEOUT
        cfg_to = self._cfg.get("timeout")
        return min(cfg_to, budget) if cfg_to else budget

    def _call(self, messages, max_tokens=None, temperature=None):
        global _nlc_seq
        _nlc_seq += 1
        seq = _nlc_seq
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = _time.monotonic()

        cfg = self._cfg
        provider = cfg.get("provider", "")
        model = cfg["model"]
        mt = max_tokens or cfg.get("max_tokens", 1024)
        api_key = cfg["api_key"]
        to = self._timeout()

        if provider == "anthropic":
            body = {"model": model, "max_tokens": mt, "messages": messages}
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=data,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            resp = _http_json(req, model, to)
            text = resp["content"][0]["text"]
            tokens_in = resp.get("usage", {}).get("input_tokens", 0)
            tokens_out = resp.get("usage", {}).get("output_tokens", 0)
        else:
            base_url = cfg.get("base_url") or "https://api.openai.com/v1"
            body = {"model": model, "max_tokens": mt, "messages": messages}
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=data,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "content-type": "application/json",
                },
            )
            resp = _http_json(req, model, to)
            text = resp["choices"][0]["message"]["content"]
            tokens_in = resp.get("usage", {}).get("prompt_tokens", 0)
            tokens_out = resp.get("usage", {}).get("completion_tokens", 0)

        duration_ms = (_time.monotonic() - t0) * 1000
        _nlc_log.append({
            "call_id": f"nlc_{seq}",
            "sequence": seq,
            "model": model,
            "system": None,
            "messages": messages,
            "response": {"role": "assistant", "content": text},
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "duration_ms": duration_ms,
            "started_at": started_at,
        })
        return text

    def complete(self, prompt, **kwargs):
        return self._call([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages, **kwargs):
        return self._call(messages, **kwargs)


class _LLMRegistry:
    def __init__(self, configs, default_timeout=None):
        self._configs = configs or {}
        self._cache = {}
        self._default_timeout = default_timeout

    def __getitem__(self, name):
        if name not in self._cache:
            cfg = self._configs.get(name)
            if cfg is None:
                raise KeyError(
                    "LLM tool '" + name + "' is not configured. "
                    "Available: " + repr(list(self._configs.keys()))
                )
            self._cache[name] = _LLMTool(name, cfg, self._default_timeout)
        return self._cache[name]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(str(e)) from e


try:
    payload = json.loads(sys.stdin.buffer.read())
    mod = types.ModuleType("_handler")
    exec(compile(payload["handler_source"], "<handler>", "exec"), mod.__dict__)
    # Inject after exec so module-level definitions in the handler cannot shadow these.
    mod.__dict__["llm"] = _LLMRegistry(payload.get("llm_configs"), payload.get("llm_timeout"))
    mod.__dict__["INPUTS_DIR"] = payload.get("inputs_dir")
    mod.__dict__["OUTPUTS_DIR"] = payload.get("outputs_dir")
    if "run" not in mod.__dict__:
        raise AttributeError("Handler must define a top-level run(args) function")
    # Redirect hardcoded /outputs/ and /inputs/ paths to the injected dirs.
    # This makes tools that use Docker-style paths work in uv mode too.
    _od = mod.__dict__["OUTPUTS_DIR"]
    _id = mod.__dict__["INPUTS_DIR"]
    if _od or _id:
        import builtins as _bi
        import os as _os
        _bi_open = _bi.open
        _os_makedirs = _os.makedirs

        def _redirect(p):
            if not isinstance(p, str):
                return p
            if _od and (p == "/outputs" or p.startswith("/outputs/")):
                suffix = p[len("/outputs"):].lstrip("/")
                return _os.path.join(_od, suffix) if suffix else _od
            if _id and (p == "/inputs" or p.startswith("/inputs/")):
                suffix = p[len("/inputs"):].lstrip("/")
                return _os.path.join(_id, suffix) if suffix else _id
            return p

        def _open_compat(file, *a, **kw):
            return _bi_open(_redirect(file), *a, **kw)

        def _makedirs_compat(name, *a, **kw):
            return _os_makedirs(_redirect(name), *a, **kw)

        mod.__dict__["open"] = _open_compat
        _os.makedirs = _makedirs_compat
    result = mod.run(payload["args"])
    captured = sys.stdout.getvalue()
    sys.stdout = _out
    if captured:
        sys.stderr.write(captured)
        sys.stderr.flush()
    try:
        s = json.loads(json.dumps(result))
    except (TypeError, ValueError):
        s = str(result)
    print(json.dumps({"output": s, "error": None, "nested_llm_calls": _nlc_log}), file=_out, flush=True)
except Exception as _e:
    captured = sys.stdout.getvalue() if hasattr(sys.stdout, "getvalue") else ""
    sys.stdout = _out
    if captured:
        sys.stderr.write(captured)
        sys.stderr.flush()
    print(
        json.dumps({
            "output": None,
            "error": {
                "type": type(_e).__name__,
                "message": str(_e),
                "traceback": traceback.format_exc(),
            },
            "nested_llm_calls": _nlc_log,
        }),
        file=_out,
        flush=True,
    )
    sys.exit(1)
