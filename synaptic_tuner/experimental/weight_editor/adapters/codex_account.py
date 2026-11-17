"""Account-authenticated Codex CLI ``ModelAdapter`` (experimental, unqualified by default).

Transport: one ``codex exec`` (first turn of a provider thread) or
``codex exec resume <thread_id>`` (later turns) subprocess per broker turn, using
the operator's existing ChatGPT login. No API key, no new billing path, no
model fallback, no retry.

What this adapter guarantees (and tests with a fake executable):

* argv is a list (no shell); every built-in tool switch this client version
  exposes is overridden off; user config/rules, project docs and environment
  context are excluded; the response is constrained by a JSON Schema envelope.
* exact stdout bytes are yielded line by line (newline included) on the
  ``provider`` channel; stderr chunks on ``client_stderr``; the launch record
  and the exact stdin prompt bytes on ``client_input`` *before* the process
  starts. Nothing is synthesized on the provider channel.
* any native tool item, unknown packet, error/failed turn, malformed or
  multiple final actions, truncated stream, missing ``turn.completed``,
  nonzero exit, thread-id drift, client-binary drift, a dirtied work dir, or a
  clean turn without client-reported returned model identity raises
  ``AdapterFailure`` after the bytes were yielded (so the broker retains them).
  The adapter never retries.

What it does NOT establish (see ``docs/experimental/weight-editor-codex-account-adapter.md``):
the model-visible tool catalogue, the model-visible instruction set, the
server-returned model, and the absence of client-internal transport retries
are not observable from ``codex exec --json``. ``mode="probe"`` runs and lists
these as blockers; ``mode="qualified"`` refuses to start without an operator
qualification record bound to the exact client bytes and effective config.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..canon import canonical_bytes, digest_of, sha256_hex, strict_loads
from ..errors import AdapterCancelled, AdapterFailure, AdapterTimeout, PortError, UnsupportedCapability
from ..ports import Executor, ModelAdapter, PortContext, PortSet, TransportEvent, TurnInput

ADAPTER_NAME = "codex_account"
ADAPTER_VERSION = "0.1.0-experimental"
PROVIDER = "openai-codex-chatgpt-account"
LAUNCH_SCHEMA = "codex_account_launch/v1"
TURN_SCHEMA = "codex_account_turn/v1"
QUALIFICATION_SCHEMA = "codex_account_qualification/v1"

# requested setting name -> codex config key. Anything else is refused (no silent drop).
SETTINGS_MAP = {
    "reasoning_effort": "model_reasoning_effort",
    "reasoning_summary": "model_reasoning_summary",
    "verbosity": "model_verbosity",
}
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_THREAD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

# Union of (a) the source paper's harness config and (b) the override list codex 0.166.0
# itself applies to its internal tool-free "temporary structured thread".
DISABLED_FEATURES = (
    "apps", "browser_use", "browser_use_external", "browser_use_full_cdp_access", "code_mode",
    "code_mode_host", "code_mode_only", "collaboration_modes", "computer_use", "context_management",
    "current_time_reminder", "deferred_executor", "enable_fanout", "goals", "hooks", "image_generation",
    "in_app_browser", "memories", "multi_agent", "multi_agent_v2", "plugins", "remote_plugin",
    "request_permissions_tool", "shell_snapshot", "shell_tool", "skill_mcp_dependency_install",
    "skill_search", "sleep_tool", "standalone_web_search", "token_budget", "tool_call_mcp_elicitation",
    "tool_suggest", "unified_exec", "unified_exec_tty", "view_image", "workspace_dependencies", "worktrees",
)

FIXED_OVERRIDES = (
    ("model_provider", '"openai"'),
    ("forced_login_method", '"chatgpt"'),
    ("approval_policy", '"never"'),
    ("sandbox_mode", '"read-only"'),
    ("web_search", '"disabled"'),
    ("project_doc_max_bytes", "0"),
    ("project_doc_fallback_filenames", "[]"),
    ("include_permissions_instructions", "false"),
    ("include_apps_instructions", "false"),
    ("include_collaboration_mode_instructions", "false"),
    ("include_environment_context", "false"),
    ("include_environment_context_time", "false"),
    ("hide_agent_reasoning", "false"),
    ("show_raw_agent_reasoning", "false"),
    ("notify", "[]"),
    ("mcp_servers", "{}"),
    ("cloud.skills.enabled", "false"),
    ("skills.include_instructions", "false"),
    ("tools.update_plan.enabled", "false"),
    ("tools.experimental_request_user_input.enabled", "false"),
)

ENV_PASS = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
            "SSL_CERT_FILE", "SSL_CERT_DIR", "CODEX_HOME")
ENV_FIXED = {"NO_COLOR": "1"}
_FORBIDDEN_ENV = re.compile(r"^(OPENAI|CODEX|AZURE_OPENAI|CHATGPT)_[A-Z0-9_]*(KEY|TOKEN|BASE_URL|API_BASE|ORG_ID|ORGANIZATION|PROJECT)$")

ANCESTOR_MARKERS = (".git", "AGENTS.md", "AGENTS.override.md", ".codex", ".agents")
MANAGED_PATHS = ("/etc/codex/managed_config.toml", "/etc/codex/config.toml", "/etc/codex/requirements.toml")

FORBIDDEN_ITEMS = frozenset({"command_execution", "file_change", "mcp_tool_call", "web_search",
                             "todo_list", "collab_tool_call"})
ENVELOPE_KEYS = ("action", "operation", "arguments_json", "choice", "proposal", "rationale")

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(ENVELOPE_KEYS),
    "properties": {
        "action": {"type": "string", "enum": ["request", "final_choice"]},
        "operation": {"type": ["string", "null"]},
        "arguments_json": {"type": ["string", "null"]},
        "choice": {"type": ["string", "null"], "enum": ["patch", "no_edit", None]},
        "proposal": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
}

NEUTRAL_INSTRUCTIONS = (
    "You are the editor model in a brokered research session. Each user message is a JSON envelope "
    "of messages from the broker. You have no native tools. Act only through your final reply, which "
    "must be one JSON object matching the response schema: use action \"request\" with operation and "
    "arguments_json (a JSON object encoded as a string) to ask the broker for one operation, or action "
    "\"final_choice\" with choice and proposal to end discovery. Put your visible justification in "
    "rationale. Unused fields are null.\n"
)


# ------------------------------------------------------------------ helpers


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(code: str, message: str) -> AdapterFailure:
    return AdapterFailure(message, code=code)


def parse_envelope(text: str) -> dict:
    """Map the agent's final structured message to one broker event. Raise ValueError if malformed."""
    doc = strict_loads(text)
    if not isinstance(doc, dict) or tuple(sorted(doc)) != tuple(sorted(ENVELOPE_KEYS)):
        raise ValueError("final message is not the exact action envelope")
    rationale = doc["rationale"]
    if not isinstance(rationale, str):
        raise ValueError("rationale must be a string")
    if doc["action"] == "request":
        op, args_json = doc["operation"], doc["arguments_json"]
        if not isinstance(op, str) or not op or doc["choice"] is not None or doc["proposal"] is not None:
            raise ValueError("request envelope needs operation and null choice/proposal")
        if not isinstance(args_json, str):
            raise ValueError("request envelope needs arguments_json")
        args = strict_loads(args_json)
        if not isinstance(args, dict):
            raise ValueError("arguments_json must encode a JSON object")
        return {"kind": "request", "operation": op, "arguments": args, "rationale": rationale}
    if doc["action"] == "final_choice":
        if doc["operation"] is not None or doc["arguments_json"] is not None:
            raise ValueError("final_choice envelope needs null operation/arguments_json")
        choice, proposal = doc["choice"], doc["proposal"]
        if choice not in ("patch", "no_edit") or not (proposal is None or isinstance(proposal, str)):
            raise ValueError("final_choice envelope needs choice patch|no_edit")
        out = {"kind": "final_choice", "choice": choice, "rationale": rationale}
        if proposal is not None:
            out["proposal"] = proposal
        return out
    raise ValueError("unknown action")


def _returned_model(ev: Mapping[str, Any]) -> str | None:
    """Only a model string the client itself put in the event; never the requested value."""
    for holder in (ev, ev.get("item") if isinstance(ev.get("item"), dict) else {}):
        m = holder.get("model")
        if isinstance(m, str) and m:
            return m
    return None


def normalize(raw: bytes) -> dict:
    """Stateless mapping of one exact stdout JSONL line (newline included) to a broker event."""
    ev = strict_loads(raw)
    if not isinstance(ev, dict) or not isinstance(ev.get("type"), str):
        raise ValueError("provider line is not a typed JSON object")
    t = ev["type"]
    model = _returned_model(ev)
    extra = {"model": model} if model else {}
    if t == "thread.started":
        tid = ev.get("thread_id")
        if not isinstance(tid, str) or not _THREAD_RE.match(tid):
            raise ValueError("thread.started without a valid thread_id")
        return {"kind": "session", "event": t, "session_id": tid, **extra}
    if t == "turn.started":
        return {"kind": "session", "event": t, **extra}
    if t == "turn.completed":
        usage = ev.get("usage")
        return {"kind": "finish", "status": "completed", "usage": usage if isinstance(usage, dict) else None,
                **extra}
    if t == "turn.failed":
        return {"kind": "finish", "status": "failed", "error": ev.get("error"), **extra}
    if t == "error":
        return {"kind": "finish", "status": "error", "message": ev.get("message"), **extra}
    if t in ("item.started", "item.updated", "item.completed"):
        item = ev.get("item")
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise ValueError("item event without a typed item")
        it, iid = item["type"], item.get("id")
        if it == "agent_message":
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("agent_message without text")
            if t != "item.completed":
                return {"kind": "message", "event": t, "item_id": iid, "text": text, "partial": True, **extra}
            if text.lstrip().startswith("{"):
                # raises on malformed structured output; payload keys are exactly what admission
                # accepts, so a client-reported model on this line is not merged here (the original
                # text is preserved by the broker's raw reference)
                return parse_envelope(text)
            return {"kind": "message", "event": t, "item_id": iid, "text": text, **extra}
        if it == "reasoning":
            return {"kind": "reasoning_summary", "event": t, "item_id": iid, "text": item.get("text"),
                    "source": "client_exposed_summary", **extra}
        if it in FORBIDDEN_ITEMS:
            return {"kind": "unknown", "native_tool_item": it, "event": t, "item_id": iid}
        return {"kind": "unknown", "unsupported_item": it, "event": t}
    return {"kind": "unknown", "unsupported_type": t}


# ------------------------------------------------------------------ adapter


class CodexAccountAdapter(ModelAdapter):
    """Trusted operator port. Construct directly or via :func:`build_codex_account_ports`.

    Parameters are all explicit; nothing is chosen by a model request.

    ``codex_path``/``expected_sha256``/``expected_version`` pin the client.
    ``work_dir`` is an empty private cwd that must stay empty; ``control_dir``
    holds the adapter-owned instructions and response schema (create-only, or
    byte-identical if present). ``mode`` is ``"probe"`` (runs, never claims
    qualification) or ``"qualified"`` (requires ``qualification_record``).
    ``turn_timeout_s`` is an optional operator wall-clock bound (no default).
    ``base_env`` defaults to ``os.environ``; only allowlisted names are passed.
    ``acknowledged_markers`` lists exact ancestor marker paths the operator accepts.
    """

    def __init__(self, *, model: Mapping[str, Any], codex_path: str | os.PathLike, expected_sha256: str,
                 expected_version: str, work_dir: str | os.PathLike, control_dir: str | os.PathLike,
                 mode: str, qualification_record: str | os.PathLike | None = None,
                 instructions_text: str = NEUTRAL_INSTRUCTIONS, turn_timeout_s: float | None = None,
                 base_env: Mapping[str, str] | None = None, acknowledged_markers: tuple[str, ...] = (),
                 max_line_bytes: int = 64 << 20, kill_grace_s: float = 2.0, verify_login: bool = True):
        if mode not in ("probe", "qualified"):
            raise PortError("mode must be 'probe' or 'qualified'; there is no default")
        self._requested = self._check_model(model)
        self.mode = mode
        self._timeout = turn_timeout_s
        if turn_timeout_s is not None and not (isinstance(turn_timeout_s, (int, float)) and turn_timeout_s > 0):
            raise PortError("turn_timeout_s must be positive or None")
        self._max_line = int(max_line_bytes)
        self._grace = float(kill_grace_s)

        self._codex = Path(codex_path).resolve(strict=True)
        actual = _sha256_file(self._codex)
        if actual != expected_sha256:
            raise PortError("codex client bytes do not match the pinned sha256", code="client_drift")
        self._client_sha = actual

        env_in = dict(os.environ if base_env is None else base_env)
        bad = sorted(k for k, v in env_in.items() if v and _FORBIDDEN_ENV.match(k))
        if bad:
            raise PortError(f"API-key/billing environment present ({', '.join(bad)}); account login only",
                            code="api_key_environment")
        self._env = {k: env_in[k] for k in ENV_PASS if k in env_in}
        self._env.update(ENV_FIXED)
        home = self._env.get("HOME")
        if not home:
            raise PortError("HOME is required for the operator's existing codex login")
        self._codex_home = Path(self._env.get("CODEX_HOME") or Path(home) / ".codex")

        self._work = Path(work_dir).resolve(strict=True)
        self._control = Path(control_dir).resolve(strict=True)
        if self._work == self._control or self._control.is_relative_to(self._work):
            raise PortError("control_dir must be outside work_dir")
        if any(self._work.iterdir()):
            raise PortError("work_dir must be empty")
        self._markers = self._ancestor_markers(Path(home))
        unack = sorted(set(self._markers) - set(acknowledged_markers))
        if unack:
            raise PortError(f"work_dir ancestors contain instruction/config markers {unack}; "
                            "acknowledge exact paths explicitly or choose another work_dir",
                            code="work_dir_markers")

        self._instructions = instructions_text.encode("utf-8")
        self._schema = canonical_bytes(RESPONSE_SCHEMA) + b"\n"
        self._instr_path = self._control / "instructions.md"
        self._schema_path = self._control / "response-schema.json"
        for path, data in ((self._instr_path, self._instructions), (self._schema_path, self._schema)):
            self._install(path, data)

        self._observed = self._observe_environment(Path(home))
        self._version = self._run_probe(["--version"])
        if self._version != expected_version:
            raise PortError("codex client version does not match the pinned version", code="client_drift")
        if verify_login:
            status = self._run_probe(["login", "status"])
            # only a boolean is retained; the status text is not stored or logged
            if "chatgpt" not in status.lower() or "api key" in status.lower():
                raise PortError("codex is not logged in with a ChatGPT account", code="login_mode")
        self._login_verified = bool(verify_login)

        self._effective = self._effective_config()
        self._effective_sha = digest_of(self._effective)
        self._record_sha = None
        if mode == "qualified":
            self._record_sha = self._check_record(qualification_record)
        elif qualification_record is not None:
            raise PortError("qualification_record is only meaningful in qualified mode")

        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    # ---------------------------------------------------------- construction

    @staticmethod
    def _check_model(model: Mapping[str, Any]) -> dict:
        try:
            provider, name, settings = model["provider"], model["model"], model["settings"]
        except (KeyError, TypeError) as exc:
            raise PortError("model block needs provider, model, settings") from exc
        if provider != PROVIDER:
            raise UnsupportedCapability(f"provider must be {PROVIDER!r}; no other route is implemented")
        if not isinstance(name, str) or not _TOKEN_RE.match(name):
            raise PortError("model must be an explicit model id")
        if not isinstance(settings, dict):
            raise PortError("settings must be an object")
        unknown = sorted(set(settings) - set(SETTINGS_MAP))
        if unknown:
            raise UnsupportedCapability(f"unsupported settings {unknown}; refusing to drop them silently")
        for k, v in settings.items():
            if not isinstance(v, str) or not _TOKEN_RE.match(v):
                raise PortError(f"setting {k} must be a plain token string")
        return {"provider": provider, "model": name, "settings": dict(settings)}

    def _ancestor_markers(self, home: Path) -> list[str]:
        user_roots = {self._codex_home.resolve() if self._codex_home.exists() else self._codex_home,
                      (home / ".agents")}
        found = []
        for d in (self._work, *self._work.parents):
            for name in ANCESTOR_MARKERS:
                p = d / name
                if (p.exists() or p.is_symlink()) and p not in user_roots and p.resolve() not in user_roots:
                    found.append(str(p))
        return found

    def _observe_environment(self, home: Path) -> dict:
        """Existence only; no file under CODEX_HOME is read."""
        ch = self._codex_home
        return {
            "codex_home_agents_md": (ch / "AGENTS.md").exists(),
            "codex_home_agents_override_md": (ch / "AGENTS.override.md").exists(),
            "codex_home_skills_dir": (ch / "skills").is_dir(),
            "codex_home_plugins_dir": (ch / "plugins").is_dir(),
            "home_agents_dir": (home / ".agents").is_dir(),
            "managed_config": sorted(p for p in MANAGED_PATHS if Path(p).exists()),
            "acknowledged_ancestor_markers": sorted(self._markers),
        }

    @staticmethod
    def _install(path: Path, data: bytes) -> None:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise PortError(f"{path.name} in control_dir differs from the adapter's bytes", code="control_drift")
            return
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _run_probe(self, args: list[str]) -> str:
        try:
            cp = subprocess.run([str(self._codex), *args], cwd=self._work, env=self._env, capture_output=True,
                                timeout=60, stdin=subprocess.DEVNULL, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PortError(f"codex {' '.join(args)} failed: {type(exc).__name__}") from exc
        if cp.returncode != 0:
            raise PortError(f"codex {' '.join(args)} exited {cp.returncode}")
        return (cp.stdout + cp.stderr).decode("utf-8", errors="replace").strip()

    def _overrides(self) -> list[tuple[str, str]]:
        ov = list(FIXED_OVERRIDES)
        ov.insert(5, ("model_instructions_file", json.dumps(str(self._instr_path))))
        ov += [(f"features.{f}", "false") for f in DISABLED_FEATURES]
        ov += [(SETTINGS_MAP[k], json.dumps(v)) for k, v in sorted(self._requested["settings"].items())]
        return ov

    def _common_args(self) -> list[str]:
        out = ["--json", "--ignore-user-config", "--ignore-rules", "--strict-config", "--skip-git-repo-check",
               "--output-schema", str(self._schema_path), "-m", self._requested["model"]]
        for k, v in self._overrides():
            out += ["-c", f"{k}={v}"]
        return out

    def argv_for(self, session_id: str | None) -> list[str]:
        if session_id is None:
            return [str(self._codex), "exec", *self._common_args(), "-"]
        return [str(self._codex), "exec", "resume", *self._common_args(), session_id, "-"]

    def _effective_config(self) -> dict:
        return {
            "argv_new": self.argv_for(None),
            "argv_resume": self.argv_for("<thread_id>"),
            "cwd": str(self._work),
            "env_names": sorted(self._env),
            "env_fixed": dict(ENV_FIXED),
            "instructions_sha256": sha256_hex(self._instructions),
            "schema_sha256": sha256_hex(self._schema),
            "client_sha256": self._client_sha,
            "client_version": self._version,
            "environment_observed": self._observed,
        }

    def _check_record(self, path) -> str:
        if path is None:
            raise PortError("qualified mode requires a qualification_record", code="adapter_unqualified")
        blob = Path(path).read_bytes()
        rec = strict_loads(blob)
        need = {
            "schema": QUALIFICATION_SCHEMA, "client_sha256": self._client_sha,
            "effective_config_sha256": self._effective_sha, "model": self._requested,
        }
        for k, v in need.items():
            if not isinstance(rec, dict) or rec.get(k) != v:
                raise PortError(f"qualification record field {k} does not match this adapter",
                                code="adapter_unqualified")
        ver = rec.get("verified", {})
        for k in ("native_tool_catalogue_empty", "model_visible_instructions", "returned_model_reported",
                  "no_client_internal_retry"):
            if not isinstance(ver, dict) or ver.get(k) is not True:
                raise PortError(f"qualification record does not verify {k}", code="adapter_unqualified")
        if not isinstance(rec.get("evidence_sha256"), str):
            raise PortError("qualification record lacks evidence_sha256", code="adapter_unqualified")
        return sha256_hex(blob)

    def blockers(self) -> list[str]:
        if self.mode == "qualified":
            return []
        out = [
            "native_tool_catalogue_unobserved: codex exec --json does not expose the model-visible tool list",
            "model_visible_instructions_unobserved: final prompt assembly (incl. CODEX_HOME AGENTS.md/skills) "
            "is not exported by codex exec",
            "returned_model_unreported: codex exec JSONL carries no server-returned model; turns fail closed",
            "client_internal_retry_unobservable: codex stream/request retries are client-internal; only "
            "surfaced error events are detected",
            "strict_config_override_validation_unverified: --strict-config is documented for config.toml, "
            "not -c overrides",
        ]
        obs = self._observed
        if obs["codex_home_agents_md"] or obs["codex_home_agents_override_md"]:
            out.append("codex_home_agents_md_present: global instructions may be injected despite "
                       "--ignore-user-config")
        if obs["managed_config"]:
            out.append("managed_config_present: " + ",".join(obs["managed_config"]))
        return out

    # ---------------------------------------------------------- port API

    def identity(self) -> dict:
        qualified = self.mode == "qualified"
        return {
            **self._requested,
            "client_version": self._version,
            "client_sha256": self._client_sha,
            "adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION},
            "effective_config_sha256": self._effective_sha,
            "effective_config": self._effective,
            "auth": {"mode": "chatgpt_account", "forced_login_method": "chatgpt",
                     "login_status_checked": self._login_verified, "api_key_environment": "rejected"},
            "returned_identity": {"source": "client_reported_only", "synthesized": False},
            "qualification": {"mode": self.mode, "qualified": qualified, "record_sha256": self._record_sha,
                              "blockers": self.blockers()},
            "capabilities": {"raw_events": True, "reasoning_summaries": True, "captures_hidden_reasoning": False,
                             "tool_isolation": "config_disabled_qualified_by_record" if qualified
                             else "config_disabled_unqualified"},
        }

    def parse_event(self, raw: bytes) -> dict:
        return normalize(raw)

    def cancel(self) -> None:
        """Thread-safe: stop the current turn and kill its process group."""
        self._cancel.set()
        with self._lock:
            proc = self._proc
        if proc is not None:
            self._kill_group(proc)

    def close(self) -> None:
        self.cancel()

    def _kill_group(self, proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # children that ignored TERM; group is owned by this turn
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            pass

    def _preflight(self, turn: TurnInput) -> str | None:
        if _sha256_file(self._codex) != self._client_sha:
            raise _fail("client_drift", "codex client bytes changed since identity was bound")
        for path, data in ((self._instr_path, self._instructions), (self._schema_path, self._schema)):
            if path.is_symlink() or path.read_bytes() != data:
                raise _fail("control_drift", f"{path.name} changed since identity was bound")
        if any(self._work.iterdir()):
            raise _fail("work_dir_dirty", "work_dir is not empty")
        sid = turn.session_id
        if sid is not None and (not isinstance(sid, str) or not _THREAD_RE.match(sid)):
            raise _fail("session_invalid", "session id is not a codex thread id")
        if turn.resumed and sid is not None:
            raise _fail("resume_after_interruption",
                        "provider thread may hold an interrupted turn; explicit operator handling required")
        if sid is None and not turn.messages:
            raise _fail("missing_bootstrap", "new provider thread without bootstrap messages")
        return sid

    def run_turn(self, turn: TurnInput) -> Iterator[bytes | TransportEvent]:
        self._cancel.clear()
        sid = self._preflight(turn)
        argv = self.argv_for(sid)
        stdin = canonical_bytes({"schema": TURN_SCHEMA, "turn_id": turn.turn_id, "turn_index": turn.turn_index,
                                 "messages": [dict(m) for m in turn.messages]}) + b"\n"
        launch = canonical_bytes({
            "schema": LAUNCH_SCHEMA, "turn_id": turn.turn_id, "argv": argv, "cwd": str(self._work),
            "env_names": sorted(self._env), "stdin_sha256": sha256_hex(stdin), "stdin_size": len(stdin),
            "client_sha256": self._client_sha, "effective_config_sha256": self._effective_sha,
            "resume_thread_id": sid,
        }) + b"\n"
        yield TransportEvent(launch, "client_input")  # durable before the process exists
        yield TransportEvent(stdin, "client_input")

        check = _TurnCheck(expected_thread=sid)
        try:
            proc = subprocess.Popen(argv, cwd=self._work, env=self._env, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                                    close_fds=True, shell=False)
        except OSError as exc:
            raise _fail("spawn_failed", f"cannot start codex: {type(exc).__name__}") from exc
        with self._lock:
            self._proc = proc
        writer = threading.Thread(target=_feed, args=(proc.stdin, stdin), daemon=True)
        writer.start()
        deadline = None if self._timeout is None else time.monotonic() + self._timeout
        killed_for: AdapterFailure | None = None
        sel = selectors.DefaultSelector()
        try:
            for f, ch in ((proc.stdout, "provider"), (proc.stderr, "client_stderr")):
                os.set_blocking(f.fileno(), False)
                sel.register(f, selectors.EVENT_READ, ch)
            buf = bytearray()
            open_streams = 2
            while open_streams:
                if self._cancel.is_set() and killed_for is None:
                    killed_for = AdapterCancelled("turn cancelled by operator")
                    self._kill_group(proc)
                if deadline is not None and killed_for is None and time.monotonic() > deadline:
                    killed_for = AdapterTimeout("operator turn_timeout_s exceeded")
                    self._kill_group(proc)
                for key, _ in sel.select(timeout=0.1):
                    try:
                        chunk = os.read(key.fileobj.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        sel.unregister(key.fileobj)
                        open_streams -= 1
                        continue
                    if key.data == "client_stderr":
                        yield TransportEvent(chunk, "client_stderr")
                        continue
                    buf += chunk
                    while True:
                        nl = buf.find(b"\n")
                        if nl < 0:
                            break
                        line = bytes(buf[: nl + 1])
                        del buf[: nl + 1]
                        yield TransportEvent(line, "provider")
                        problem = check.feed(line)
                        if problem is not None and killed_for is None:
                            killed_for = problem
                            self._kill_group(proc)  # stop before a native tool or client retry proceeds
                    if len(buf) > self._max_line and killed_for is None:
                        yield TransportEvent(bytes(buf), "provider")
                        buf.clear()
                        killed_for = _fail("line_too_long", "provider line exceeded max_line_bytes")
                        self._kill_group(proc)
            if buf:
                yield TransportEvent(bytes(buf), "provider")  # exact partial bytes, then fail
                if killed_for is None:
                    killed_for = _fail("truncated_stream", "provider stream ended mid-line")
            rc = proc.wait()
            if self._cancel.is_set() and killed_for is None:  # killed by cancel() from another thread
                killed_for = AdapterCancelled("turn cancelled by operator")
        finally:
            sel.close()
            if proc.poll() is None:
                self._kill_group(proc)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # stray children of a finished client
                except (ProcessLookupError, PermissionError):
                    pass
            for f in (proc.stdout, proc.stderr):
                try:
                    f.close()
                except OSError:
                    pass
            with self._lock:
                self._proc = None
        if killed_for is not None:
            raise killed_for
        if rc != 0:
            raise _fail("nonzero_exit", f"codex exited {rc}")
        final = check.finish()
        if final is not None:
            raise final
        if any(self._work.iterdir()):
            raise _fail("work_dir_dirty", "work_dir changed during the turn")


def _feed(pipe, data: bytes) -> None:
    try:
        pipe.write(data)
        pipe.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


class _TurnCheck:
    """Per-turn protocol state over the same normalization the broker sees."""

    def __init__(self, expected_thread: str | None):
        self.expected = expected_thread
        self.lines = 0
        self.thread: str | None = None
        self.turn_started = 0
        self.completed = False
        self.actions = 0
        self.last_agent_was_action = False
        self.models: set[str] = set()

    def feed(self, line: bytes) -> AdapterFailure | None:
        self.lines += 1
        if self.completed:
            return _fail("output_after_completion", "provider output after turn.completed")
        try:
            ev = normalize(line)
        except Exception as exc:  # malformed JSON or malformed structured action
            return _fail("malformed_provider_event", f"unparseable provider line: {type(exc).__name__}")
        if isinstance(ev.get("model"), str):  # only what the broker will also see normalized
            self.models.add(ev["model"])
        kind = ev["kind"]
        if kind == "unknown":
            if "native_tool_item" in ev:
                return _fail("native_tool_event", f"client emitted native tool item {ev['native_tool_item']}")
            return _fail("unknown_provider_event", "unsupported provider packet")
        if self.lines == 1 and ev.get("event") != "thread.started":
            return _fail("missing_thread_start", "first provider event is not thread.started")
        if ev.get("event") == "thread.started":
            if self.thread is not None:
                return _fail("session_drift", "second thread.started in one turn")
            self.thread = ev["session_id"]
            if self.expected is not None and self.thread != self.expected:
                return _fail("session_drift", "resumed thread id differs from the journaled session")
        elif ev.get("event") == "turn.started":
            self.turn_started += 1
            if self.turn_started > 1:
                return _fail("multiple_turns", "more than one turn.started")
        elif kind == "finish":
            if ev["status"] != "completed":
                return _fail("provider_error", f"client reported {ev['status']}")
            if self.turn_started != 1:
                return _fail("unbalanced_turn", "turn.completed without turn.started")
            self.completed = True
        elif kind in ("request", "final_choice"):
            self.actions += 1
            self.last_agent_was_action = True
            if self.actions > 1:
                return _fail("multiple_actions", "more than one structured action in one turn")
        elif kind == "message" and not ev.get("partial"):
            self.last_agent_was_action = False
        return None

    def finish(self) -> AdapterFailure | None:
        if not self.completed:
            return _fail("missing_turn_completion", "stream ended without turn.completed")
        if self.actions != 1 or not self.last_agent_was_action:
            return _fail("no_structured_action", "turn did not end with exactly one structured action")
        if not self.models:
            return _fail("returned_model_unobserved",
                         "the client reported no returned model on this turn; nothing is synthesized")
        return None


# ------------------------------------------------------------------ composition


def build_codex_account_ports(ctx: PortContext, executor: Executor, **adapter_kwargs: Any) -> PortSet:
    """Compose the adapter with a caller-supplied real ``Executor``.

    Model identity comes only from the config's model block; ``adapter_kwargs``
    are operator settings (paths, pins, mode). No executor is invented here.
    """
    if not isinstance(executor, Executor):
        raise PortError("an Executor instance must be supplied by the caller")
    if "model" in adapter_kwargs:
        raise PortError("model comes from the config's model block only")
    return PortSet(CodexAccountAdapter(model=ctx.model, **adapter_kwargs), executor)
