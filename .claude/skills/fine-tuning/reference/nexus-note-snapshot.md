# Nexus Note Snapshot

`snapshot_nexus_notes.py` creates a deterministic JSONL snapshot of only the notes explicitly listed in its YAML config. It is read-only: every Nexus invocation is a `content read` at line 1; it never lists, searches, or calls write-capable Nexus tools.

```bash
python .skills/fine-tuning/scripts/snapshot_nexus_notes.py \
  --config private/nexus_note_snapshot.yaml
```

The configuration requires a Nexus `vault`, `workspace`, and `session`, ordered note entries (`id`, `path`, optional non-secret `metadata`), an explicit `output_path`, and optional `strip_frontmatter`. Output is private source material and must be stored outside Git.

The adapter validates the nested Nexus JSON envelope, requires successful reads for the configured path, and reconstructs only strictly sequential `1: `, `2: `… content lines. It fails closed on malformed envelopes, path mismatches, line gaps, or malformed front matter. Each JSONL row contains the selected text plus full and selected SHA-256 hashes.

On Windows, the common `nexus.cmd` shim is not passed to `CreateProcess` directly. The adapter resolves `nexus` and `node` with `shutil.which`, verifies a supported `nexus.cmd`/`nexus.bat` shim with its sibling `nexus-cli.js`, then invokes Node with an argument list. It never enables `shell=True` or interprets shim text. Missing runtimes and unsupported shims fail with stable machine codes.

Writes are batch-atomic: notes are all read and validated before an artifact is created. Existing identical bytes are idempotent success; differing bytes fail with `OUTPUT_COLLISION`. Stdout is a small JSON status object only; failures use stable error codes without note prose.
