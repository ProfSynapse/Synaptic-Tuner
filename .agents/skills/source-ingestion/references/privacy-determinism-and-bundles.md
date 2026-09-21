# Privacy, determinism, and bundles

Context: read when checking execution evidence or deciding whether a normalized
bundle is safe to hand off.

## Key idea

Operational runs may differ while normalized semantic bytes remain identical.
Trust explicit public verification and semantic bundle identity, not a path or a
run ID alone.

Effective admission limits are operational authority and therefore affect
authority identity. They are deliberately excluded from semantic bundle bytes:
identical admitted source bytes under the same structure produce the same
bundle even when their admission budgets differ.

Parsing profile is semantic structure authority, not an admission limit. A
profile-version change therefore changes structure and bundle identity even
when source bytes are identical; repeated runs under the same profile remain
byte-deterministic.

## Private by default

Configs and outputs containing corpus details belong in a private, ignored, or
otherwise protected location. Public lifecycle material excludes source content,
metadata values, absolute paths, and exception prose. Logical paths use selection
aliases plus safe relative POSIX paths.

## Bundle contract

A valid V1 bundle directory has exactly:

- `manifest.json`: canonical semantic manifest, including structure-set and
  aggregate content identities.
- `items.jsonl`: canonical normalized items sorted by logical path.

The content-addressed `bundle_ref` is `bundle-` plus the bundle digest. The
manifest binds exact structure-set content, item bytes, logical paths, and item
identities. It deliberately excludes operational data such as request ID, run
ID, plan fingerprint, timestamp, and host path.

Do not duplicate verification logic in a skill script. The public runtime owns
bounded reads, identity checks, exact inventory, canonical encodings, digest
recalculation, and tamper detection. A later tamper is detected only by an
explicit verification call; historical `show` or `result` records do not reopen
the bundle.

## Deterministic repetition

For independently admitted identical source bytes and identical semantic
structure config, expect:

- byte-identical `manifest.json` and `items.jsonl`;
- identical `bundle_digest` and `structure_set_digest`;
- different snapshot, plan, run, authority, and outcome identities.

This distinction prevents ephemeral execution context from contaminating the
dataset source identity while keeping each operation auditable.

## Publication uncertainty

If final publication durability or verification is uncertain, the runtime uses
`reconcile_required` with `effect_uncertain`. Reconcile the exact retained
semantic identity through the public lifecycle. Do not publish again, delete an
orphan stage, or guess whether a path is authoritative.
