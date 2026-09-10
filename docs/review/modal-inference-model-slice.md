# Modal inference model preparation slice

`ModalPinnedModelPreparer` adapts the existing, pinned
`prepare_model_snapshot` implementation to the generic inference serving
contract. It calls that loader exactly once and introduces no downloader,
cache, provider authority, or credential callback.

The adapter owns three explicit private execution roots and a redacted token.
It requires the loader result to remain a canonical directory below the exact
destination root, rejects redirected ancestors, and delegates the bounded,
no-follow inventory capture to `capture_pinned_model_snapshot`. It then checks
the returned generic snapshot's model, revision, root, and relative snapshot
identity.

Inventory capture is local integrity evidence, not upstream authentication.
The existing loader remains responsible for authenticated provider metadata
and bytes. The destination root must be consumer-owned and inaccessible to
concurrent writers through the complete preparation and inference lifetime.
No public registration or inference runtime activation is part of this slice.
