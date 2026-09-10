# SFT archive stream slice

This bounded prerequisite moves the existing SFT archive semantic validator to
one seekable binary-stream boundary. The runtime verifier explicitly adapts its
already-authenticated byte values with `io.BytesIO`; a later retrieved-model
materializer can instead supply a private staged regular file without first
copying an archive of up to 64 GiB into another bytes object.

The refactor does not change artifact authority, verification receipts,
accepted archive formats, model or tokenizer member rules, safetensors checks,
or any size limits. It does not extract archives or introduce a downloader,
cache, filesystem destination, model loader, provider call, or public API.

The caller retains ownership of a stream on success and failure. The stream
must be seekable and positioned at byte zero. Tar members remain
limited to 1,024 regular, flat, canonical names; individual and aggregate
expanded-size limits remain 32 GiB and 64 GiB. JSON sidecars retain their
existing bounded complete reads, while safetensors payloads continue to be
examined in 1 MiB chunks.

This prerequisite does not yet make a safe materializer or extractor.
`TarFile.getmembers()` accumulates headers before the 1,024-member check, and
the standard tar parser can process PAX or GNU extension records before the
validator rejects their resulting member names. Tokenizer JSON sidecars also
retain their existing complete-read ceiling of 512 MiB. A future materializer
must independently bound raw tar header and extension processing and enforce
an entry budget before it extracts any member.

Focused regression coverage compares file-backed and in-memory results, proves
bounded weight reads and caller ownership, and exercises non-seekable and
failed-I/O streams plus empty, malformed, truncated, oversized, noncanonical,
unsupported, and malformed-safetensors inputs.

Integrated verification (2026-09-09 local): all 122 tests in `tests/runtime`
passed in 0.65 seconds under clean CPython 3.12.9 / pytest 8.4.2 without Modal
or system-site packages. Independent focused plus semantic-artifact review
passed 83 tests in 0.37 seconds. The Modal 97-pin and offline trainer 66-member
source inventories remain `CURRENT`; this helper is not a new remote closure
member. The earlier 2,043-test run and installed wheel predate this change.
