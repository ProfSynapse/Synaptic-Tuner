# Writing a Transcript Adapter

The engine is format-agnostic. To support a new transcript format (a different
agent CLI, a ShareGPT export, an OpenAI conversation dump, your own logger),
write one adapter. Everything downstream — labeling, secret scrub, token budget,
outcome tiering, dedup — works unchanged.

## The contract

Implement `Adapter` from `scripts/adapters/base.py`:

```python
from .base import Adapter, blocks_text

class MyFormatAdapter(Adapter):
    name = "myformat"                     # referenced by `format:` in config

    def scope_key(self, path, root):      # string matched against scope filters
        ...                               # e.g. a project slug or recorded cwd

    def project(self, path, root):        # human label for the project
        ...

    def parse(self, path):                # -> (events, signals)
        ...

    def parent_path(self, path):          # optional: nested/subagent -> parent file
        return None
```

Register it in `scripts/adapters/__init__.py`:

```python
from .myformat import MyFormatAdapter
_ADAPTERS = [ClaudeCodeAdapter(), CodexAdapter(), MyFormatAdapter()]
```

## The normalized event model

`parse(path)` returns `(events, signals)`.

`events` — ordered list, one dict per turn:

```python
{"role": "human",     "text": "..."}
{"role": "assistant", "text": "...", "tool_calls": [{"name": "Edit", "input": {...}}]}
{"role": "tool",      "tool_error": False, "command": "pytest -q", "output": "5 passed"}
```

Rules:
- One row is emitted per `assistant` event.
- A `tool` event carries the `command` it answers and the `output` text. This is
  what makes outcome detection (tests/build/commit) work **generically** — you
  don't write any outcome logic, you just surface the command + output.
- `command` should be the shell command string when the tool is a shell/exec
  tool (used by `session_outcome` regexes); `""` otherwise.

`signals` — session-level booleans the engine folds into outcomes:

```python
{"pr_created": True}   # e.g. the format logs an explicit "PR opened" event
```

Return `{}` if your format has no such signals (commits/PRs will still be
detected from commands).

## Tips

- Skip metadata/noise lines silently; only emit human/assistant/tool turns.
- If your format batches multiple tool results in one record, emit one `tool`
  event per result.
- `blocks_text()` in `base.py` handles the common "content is a string OR a list
  of typed text blocks" shape.
- For nested/subagent transcripts, return the parent session's file path from
  `parent_path()` so the child inherits the parent's outcome tier.
- Validate by running `--smoke --limit 5 --show 3` and eyeballing that prompts,
  completions, and tool calls look right before a full run.

## Reference adapters

- `adapters/claude_code.py` — event-stream JSONL, `tool_use`/`tool_result`
  blocks paired by id, `pr-link` session signal, subagent → parent inheritance.
- `adapters/codex.py` — `response_item` payloads, `function_call` /
  `function_call_output` pairs, exit status parsed from plaintext output, cwd
  scope-key from `session_meta`.
