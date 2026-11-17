# Weight editor: Codex account adapter (experimental, unqualified)

`synaptic_tuner.experimental.weight_editor.adapters.codex_account.CodexAccountAdapter`
is a `ModelAdapter` that drives the locally installed Codex CLI through the
operator's existing ChatGPT login. It is not a Tuner API v1 promise. Synthetic
fake-executable tests pass. **It has not been run against a real model and is
not qualified for any study.**

## Transport

Each broker turn launches one subprocess, using argv lists and never a shell:

- If the turn has no session, the adapter runs
  `codex exec --json --ignore-user-config --ignore-rules --strict-config --skip-git-repo-check --output-schema <control>/response-schema.json -m <model> -c ... -`.
  This also covers a core bootstrap after a handoff.
- If the turn has a session, it runs
  `codex exec resume <same flags> <thread_id> -`.

The prompt is passed on stdin as canonical JSON of the shape
`{"schema":"codex_account_turn/v1","turn_id","turn_index","messages":[...]}`.

`resume` does not accept `--sandbox` or `--cd`. For that reason both forms set
`sandbox_mode="read-only"` with `-c`, and the process `cwd` is the private
`work_dir`. Both forms use identical flags, and a test asserts this.

### Config overrides (`-c`)

- `model_provider="openai"`, `forced_login_method="chatgpt"`,
  `approval_policy="never"`, `sandbox_mode="read-only"`,
  `web_search="disabled"`.
- `model_instructions_file=<control>/instructions.md`. These are neutral
  protocol instructions, replaceable through `instructions_text`, and their
  bytes are bound into the identity.
- `project_doc_max_bytes=0` and `project_doc_fallback_filenames=[]`.
- `include_{permissions,apps,collaboration_mode}_instructions=false` and
  `include_environment_context[_time]=false`.
- `show_raw_agent_reasoning=false`, `hide_agent_reasoning=false`,
  `notify=[]`, `mcp_servers={}`.
- `cloud.skills.enabled=false`, `skills.include_instructions=false`,
  `tools.update_plan.enabled=false`,
  `tools.experimental_request_user_input.enabled=false`.
- `features.<f>=false` for each name in `DISABLED_FEATURES`. That list is the
  union of:
  - the source paper's `harness/codex_config.toml`, and
  - the override list codex-cli 0.166.0 applies to its own internal tool-free
    "temporary structured thread". That list was recovered from the binary's
    strings and was not confirmed by running the binary.
- Requested settings: only `reasoning_effort`, `reasoning_summary` and
  `verbosity`, mapped to `model_*` keys. Any other setting key is refused
  rather than dropped.

### Environment

Only these names are passed from `base_env` (default `os.environ`): `PATH`,
`HOME`, `USER`, `LOGNAME`, `LANG`, `LC_*`, `TZ`, `SSL_CERT_*` and `CODEX_HOME`.
`NO_COLOR=1` is added.

If any API-key or billing variable is present, construction is refused. This
covers `OPENAI_*`, `CODEX_*` and `AZURE_OPENAI_*` names ending in `KEY`,
`TOKEN`, `BASE_URL` and similar.

When `verify_login=True`, `codex login status` must report ChatGPT. Only a
boolean is retained from that check.

`HOME` and `CODEX_HOME` are never repurposed. The adapter checks for the
existence of `$CODEX_HOME/AGENTS.md`, `AGENTS.override.md`, `skills/` and
`plugins/`, `~/.agents` and `/etc/codex/*` but never reads them. Those
existence observations are bound into the effective config.

## Identity

`identity()` returns the following:

- `provider`, `model` and `settings`, exactly as in the config.
  `provider` must be `openai-codex-chatgpt-account`.
- The client's version and sha256. Both are pinned by the operator, the
  sha256 is re-checked before every turn, and drift raises `client_drift`.
- The effective config: argv templates, cwd, env names, instruction and
  schema sha256, and environment observations. Its digest is
  `effective_config_sha256`.
- Auth mode, `returned_identity: {"source":"client_reported_only","synthesized":false}`,
  the qualification mode and its blockers, and the capabilities.

The core hashes the adapter source. Any change to these values is identity
drift, and the broker refuses to resume.

## Capture

The adapter yields three kinds of event, in this order:

1. `client_input` before the process starts: the launch record (argv, cwd,
   env names, stdin sha256/size, pins, resume thread id), then the exact stdin
   bytes.
2. `provider`: every stdout JSONL line, byte-exact with its trailing newline.
   A partial final line is yielded as-is and then raises `truncated_stream`.
3. `client_stderr`: stderr chunks, byte-exact as they arrive.

Both pipes are multiplexed with `selectors`, so neither pipe can deadlock. The
stdout line buffer is the only accumulation. It is bounded by
`max_line_bytes` (default 64 MiB), which raises `line_too_long`. There is no
token or cost cap.

## Normalization (`parse_event`)

The mapping is stateless and runs again in `run_turn`, so the adapter and the
broker see the same normalized events.

| Codex event | Broker kind |
|---|---|
| `thread.started` | `session` with `session_id`. `model` is included only if the client reported one. |
| `turn.started` | `session` |
| `turn.completed` | `finish` with `status: "completed"` and `usage` |
| `turn.failed` / `error` | `finish` with status `failed` or `error`. The adapter then raises `provider_error`. |
| `item.*` of type `reasoning` | `reasoning_summary` (`source: client_exposed_summary`). Hidden reasoning is never claimed. |
| `item.completed` of type `agent_message` whose text starts with `{` | Must be the exact envelope `{action, operation, arguments_json, choice, proposal, rationale}`. It becomes `request` (`operation`, `arguments` from strict-parsed `arguments_json`, `rationale`) or `final_choice` (`choice`, `rationale`, `proposal` if any). A malformed envelope raises, and the broker records it as `unknown`. |
| Any other `agent_message` text | `message`. Prose never becomes a request. |
| `command_execution`, `file_change`, `mcp_tool_call`, `web_search`, `todo_list`, `collab_tool_call` | `unknown` with `native_tool_item` |
| Any other type | `unknown` |

The original agent text is preserved by the broker's raw reference. It is not
copied into the request payload, because admission accepts only
`operation`, `arguments`, `rationale` and `label`.

## Fail-closed checks

In every failure case below, the bytes are yielded first and `AdapterFailure`
is raised afterwards. The adapter never retries. Where possible, the process
group is killed immediately so that a native tool or a client-internal
reconnect cannot continue.

- `native_tool_event`, `unknown_provider_event`, `malformed_provider_event`
  and `provider_error`.
- `missing_thread_start` and `session_drift`. On resume, the reported thread
  id must equal the journaled session.
- `multiple_turns`, `unbalanced_turn`, `output_after_completion` and
  `missing_turn_completion`.
- `multiple_actions` and `no_structured_action`. The last agent message must
  be the one structured action.
- `truncated_stream`, `line_too_long` and `nonzero_exit`.
- `work_dir_dirty`: the private cwd must be empty before and after every turn.
- `client_drift` and `control_drift`.
- `returned_model_unobserved`: no normalized event on a clean turn carried a
  client-reported model. Nothing is synthesized from the requested model.
- These are refused before spawning:
  - `resume_after_interruption`: `TurnInput.resumed` together with a session.
  - `session_invalid`.
  - `missing_bootstrap`.
- `AdapterTimeout`, raised only when the optional `turn_timeout_s` is set.
- `AdapterCancelled`, raised by `cancel()` from any thread. Closing the
  generator also cleans up. In both cases the process group is sent SIGTERM,
  then SIGKILL after `kill_grace_s`.

## Modes and qualification

`mode` has no default.

- `probe` runs the adapter and reports `qualified: false` along with its
  blockers.
- `qualified` requires a `qualification_record` JSON with:
  - `schema: codex_account_qualification/v1`;
  - `client_sha256`, `effective_config_sha256` and `model`, matching the
    adapter;
  - `verified.{native_tool_catalogue_empty, model_visible_instructions, returned_model_reported, no_client_internal_retry}`
    all true;
  - `evidence_sha256`.

The record's sha256 is bound into the identity. The adapter checks only that
the record is consistent. It cannot verify the evidence itself.

## Composition

```python
from synaptic_tuner.experimental.weight_editor.adapters.codex_account import build_codex_account_ports
ports = build_codex_account_ports(ctx, executor=my_real_executor,
    codex_path=..., expected_sha256=..., expected_version="codex-cli 0.166.0",
    work_dir=empty_private_dir, control_dir=adapter_control_dir, mode="probe",
    acknowledged_markers=(...))
```

`ctx.model` is the only source of model identity. The caller must supply a
real `Executor`. There is no executor default and no fake.

`work_dir` ancestors that contain `.git`, `AGENTS.md`, `AGENTS.override.md`,
`.codex` or `.agents` are refused unless they are listed exactly in
`acknowledged_markers`. User-level `CODEX_HOME` and `~/.agents` are recorded
as environment observations instead. Use a `work_dir` outside any repository
for real runs.

## Unresolved gaps (block qualification)

1. **The model-visible tool catalogue is not observable.** `codex exec --json`
   does not export the tool list sent to the provider. `codex debug
   prompt-input` renders input items, not tools, and has no
   `--ignore-user-config`. Disabled flags are configuration intent, not a
   measured catalogue.
2. **The model-visible instruction set is not observable.** It is unknown
   whether `$CODEX_HOME/AGENTS.md` (present on this machine) and user skills
   are injected under `--ignore-user-config`. They are probably still loaded.
   Checking this means rendering the prompt with the owner's real
   `CODEX_HOME`, which exposes private instructions. That is the parent's
   decision.
3. **The returned model is not reported.** Codex 0.166.0 exec JSONL has no
   model field. Under the new core contract every clean turn needs observed
   returned identity, so **every real turn currently fails closed with
   `returned_model_unobserved`**. The adapter forwards a `model` field if a
   future client emits one. Other possible evidence sources (rollout files,
   response headers) are not exported by exec. Reading the session store
   would touch the owner's other sessions.
4. **Client-internal retries cannot be disabled or fully observed.**
   Stream and request retries happen inside the built-in `openai` provider.
   Only retries that surface as `error` events are detected, and the process
   is killed when one appears.
5. **`--strict-config` may not cover `-c` overrides.** Help text documents it
   for `config.toml`. A misspelled override might be ignored silently.
   `agents.enabled=false` from the paper is omitted because the key was not
   found in the 0.166.0 binary.
6. **Silent compaction is undetectable** unless it surfaces as an unknown
   item, which raises.
7. **Provider threads persist in the owner's `$CODEX_HOME/sessions`.**
   `--ephemeral` would break `resume`.
8. **The exec JSONL item shapes come from the binary's strings and documented
   behaviour, not from a live capture.** The first real probe should be
   checked for unknown events, which raise.
