# Modal coordinator generic-log codec

The internal `ModalCoordinatorLogChunk` is a new, closed
`synaptic.modal-log-chunk/v2` wire value for the Foundation-native Modal path.
It does not change, alias, or dual-parse the legacy `StructuredLogChunkV1`.

Each nonempty chunk contains exact generic `RunLogEntry` values. Parsing uses
`RunLogEntry.from_dict`, so RFC 3339 timestamps, levels, event and message text,
UTF-8 message size, and record sequence retain the public run-log contract.
The codec never invents a timestamp. Records are reconstructed into fresh
immutable values, serialized canonically, bounded by `BoundsPolicyV1`, and
committed by the exact payload SHA-256.

The v2 chain-node digest commits the generation, chunk sequence, predecessor,
payload digest, job/effect/plan identity, and invocation nonce under the new
`synaptic.modal-log-chain-node/v2` schema. `validate_modal_log_chain` accepts
only a nonempty exact tuple of exact v2 chunks. It requires one identity,
contiguous chunk sequences beginning at zero, an exact predecessor chain, and
globally contiguous generic record sequences beginning at zero. Caller bounds
limit individual canonical chunks, chunk count, per-chunk records, and total
records, so a supplied chain is not an unbounded tuple.

The chunk's `payload_digest` covers the complete serialized generic entries;
`RunLogEntry.size_bytes` continues to mean message UTF-8 bytes, not serialized
record or chunk size. A future reader may use that field for its generic query
byte budget while independently enforcing the wire and chain bounds here.

This module performs no SDK access, filesystem I/O, authentication, timestamp
synthesis, or provider calls. Existing v1 metadata may continue treating the
chunk digest as opaque; integrating production and reader code is separate.
