# Modal v1 Volume root diagnostic — 2026-09-27

The first Qwen 3.5 4B L40S training attempt stopped before an optimizer step at
`RUN_WORKER_ENTRYPOINT_MOUNT_CONTROL_LINK`. Its one-shot claim and resources are
preserved. No LoRA weights were produced by that attempt.

The checked-in no-training L40S probe later observed `LINK` for all three fresh
v1 Volume roots under both `/mnt` and `/workspace`. A marker challenge returned
`MATCH` for each root, and a subsequent exact-target diagnostic returned
`MATCH_MODAL_ID`: each link text matched
`/__modal/volumes/<the submitted Volume ID>`. These observations describe the
fresh diagnostic Functions at observation time, not a permanent provider
guarantee or proof that the failed training Function used the same inodes.

After WSL recovered, the isolated descriptor-root and exact-ID host-reader
candidates passed 25 provider-free Linux tests. Security review stopped a
proposed host listing call: Modal 1.5.4 can deserialize an entire listing
batch before an application entry cap. Listing was removed from the read-only
marker diagnostic and then from the reusable reader candidate. The revised
reader makes one retry-disabled `VolumeGetFile2` request with an explicit
deadline, accepts one signed URL for the 32-byte marker, reads at most 33
application bytes, and verifies exact size, range metadata, and SHA-256.

One read-only invocation against the retained, owner-private probe claim
returned `MARKER_MATCH` for its control-Volume marker. No GPU Function,
Volume, write, or training call was created by that check. The focused Linux
tests for the candidates and inspector passed. The claim and receipt are
owner-private local authority, not a MAC against a same-UID editor.

This qualifies only the small exact-ID read observed above. It does not
qualify bounded listing, large artifact streaming, arbitrary server range
behavior, mount-path stability during training, or the production worker.
Before another training call, the signed dispatch must bind distinct markers
to exact Volume IDs; the worker must retain authenticated nofollow roots; the
offline trainer must operate entirely in private scratch; exact artifacts and
completion must be published and committed in order; and changed locked source
must pass the normal release and CPU gates. Do not relax the existing
no-symlink guard, replay consumed claims, or infer training success from this
diagnostic.

## Production candidate checkpoint

The reviewed v2 candidate now signs distinct per-Volume marker commitments,
uploads and exact-ID reads back each create-only marker before spawn, and binds
the remote links to exact IDs and retained nofollow descriptors. The generic
trainer receives private scratch paths. Pinned model files cross into a fresh
cache only after private verification; verified output artifacts and completion
cross back through bound roots with exclusive publication and ordered commits.
The original failed attempt remains untouched.

The integrated provider-free marker/worker/host suite passed 158 Linux tests.
Security review found and prompted fixes for SDK-created 0755 descendants under
the private 0700 scratch anchor and for an unbounded host marker upload/commit.
The latter now has a 60-second local deadline and preserves the consumed claim
if a provider write finishes late. Modal 1.5.4 locally forwards
`batch_upload(force=False)` as `disallow_overwrite_existing_files=True`.
The runtime and inference locks were refreshed only for the reviewed
`model_snapshot.py` hash; both default checks are current.

The standalone fake transcript was updated to model signed v2 marker upload,
commit, and exact-ID readback. A mounted-cache regression also caught and fixed
an attempted pathname inspection of the provider symlink in bound mode.
The combined provider-free release and security suite now passes 345 Linux
tests, and both lock checks are current after the final model-source change.

This is still not GPU authority. The exact source wheel has not been built and
CPU-qualified, and no optimizer step or LoRA artifact has been verified.
Partial failed create-only writes remain as collision evidence. A floating
named Function can also be replaced by an external administrator between
reobservation and spawn; this existing provider
race is outside the caller's guarantee.
