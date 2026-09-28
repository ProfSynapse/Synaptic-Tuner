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

At that checkpoint the exact source wheel had not yet been built and
CPU-qualified. No optimizer step or LoRA artifact had been verified.
Partial failed create-only writes remain as collision evidence. A floating
named Function can also be replaced by an external administrator between
reobservation and spawn; this existing provider
race is outside the caller's guarantee.

## Exact-source CPU pass and host submit diagnosis

Commit `dc92d41cf67945b1b7b7623c14981bd630937f14` subsequently passed exact-wheel
CPU qualification and was pushed on the existing branch. That receipt proved
package/runtime admission, not GPU execution or a saved adapter. A fresh source
attempt stopped at `SOURCE_WHEEL`; variable WSL Git-status latency was observed,
but no retained exception establishes a timeout as its cause.

The next fresh attempt, `modal-97004f9fded403570b1a7ac2`, passed source preparation,
CPU qualification, and input staging, then returned `RUN_SUBMIT_RECONCILE_REQUIRED`.
Its private journal retains the submit claim, binding, and marker material but
no call-catalog row. The deployment reported zero current tasks when inspected;
neither observation proves that a call was never spawned. The claim remains
consumed and is not replayable.

Inspection of pinned Modal 1.5.4 source and the
[official Volume guide](https://modal.com/docs/guide/volumes) found two host-side
contract mismatches. Exiting `batch_upload(force=False)` sends `VolumePutFiles`
with overwrite disabled; the extra `commit()` is documented for mounted
container filesystem changes, not host batch uploads. Separately, raw marker
readback used `asyncio.run` despite the client's channel belonging to Modal's
synchronizer loop. A network-free pinned-dependency reproduction raised the
cross-loop Future error. The existing successful marker inspector already used
`synchronizer.create_blocking`. These establish a redundant operation and an
event-loop defect, but do not establish the exact historical failing call.

The existing packaged-call inspector now has a read-only `--inspect-markers`
mode. It authenticates the exact private claim, binding, and marker catalog
before bounded exact-ID reads; 59 focused POSIX tests passed. Against the failed
attempt it returned control `MATCH`, artifacts `UNAVAILABLE`, and model-cache
`UNAVAILABLE`. Thus the first marker upload succeeded. The latter two results
are inconclusive, not proof of absence. No Volume write, commit, spawn, or
attempt replay occurred during this diagnostic.

The corrective candidate removes the redundant host commit and bridges raw
reads onto Modal's loop. Provider test doubles must publish at batch-context
exit rather than requiring the same extra commit as production. Independent
review, provider-free tests, exact-source CPU qualification, and a fresh bounded
GPU attempt remain necessary; the failed claim and its partial effects stay
untouched.

## September 28 validation after storage recovery

The final provider-free validation passed 592 tests: 185 adjacent packaged
provider/worker tests and 407 combined host, standalone CLI, inspector, and
transport tests. The standalone fake reader avoids importing optional HTTP
dependencies from the test interpreter when its read method is replaced; it
still verifies publication order, exact bytes/digests, and no spawn on mismatch.
The host-composition fake now supplies the SDK loop bridge. Duplicate shadowed
test definitions were removed, and host `commit()` is an explicit test failure.
A separate network-free probe under CPython 3.11.14 and Modal 1.5.4 successfully
bridged an asynchronous operation from a worker thread through the real SDK
synchronizer. This tests the loop bridge, not live Volume behavior.

Both runtime lock checks remain CURRENT without changing locked pins or remote
source. Canonical skill copies are synchronized. The recipe plan still resolves
all 220 rows (181 train, 39 validation), the same model revision and runtime
profile, one L40S, and two optimizer steps. The live read-only L40S rate remains
USD 1.95/hour. No new CPU or GPU call is claimed by these local checks.

## September 28 source-wheel gate

The reviewed SDK correction was committed and pushed as
`02d5ce826c73e0130be65e13d3714fc833acd87e`. Fresh attempt
`modal-cc7f2aefda52a13d86f7d7c3` then stopped at
`SOURCE_WHEEL / SOURCE_STATE_INVALID`, before CPU qualification or GPU submission.
Its immutable read-only journal inspection found one build claim and no catalog
rows. The authenticated claim binds the intended commit above; the current WSL
HEAD and source root agree. The consumed attempt and resources remain preserved.
This boundary is not zero cloud effects: host provisioning creates Volumes and
Secrets and enters a build App before preparing the source wheel. No CPU
qualification or GPU training ran; no cleanup or resource reuse is authorized
by this diagnosis.

A read-only repetition of the source-check sequence passed: HEAD checks matched,
both tracked-status checks returned no stdout or stderr, and the commit archive
contained 17,530,880 bytes. Five subsequent status checks took 2.30–3.05 seconds.
An earlier slow shell invocation included WSL startup and does not prove the
internal 30-second Git timeout was exceeded. The historical failing subcheck is
not recoverable from the retained broad category; timeout remains a hypothesis,
not a diagnosis. The next small change distinguishes fixed source-check failure
categories while preserving validation, deadlines, and non-replay behavior. It
does not change the recipe and is not itself a training fix.

The diagnostic-only patch passed independent static review. The first combined
regression run had 296 passes and one failure in the existing standalone
`SFT_UNKNOWN` simulated-start case; the exact case passed in isolation and a
complete repeated run passed all 297 tests. That unexplained intermittency is
retained here, not reclassified as a source-wheel failure. Source-wheel, host,
and CLI projection tests passed in both runs. Runtime and inference locks remain
CURRENT without changed pins, and canonical skills are synchronized. These
results support a new exact-source qualification attempt, not a claim that the
historical failure is resolved or a GPU run has succeeded.

### Measured status timeout on the diagnostic commit

Fresh attempt `modal-b8a4284c42e45ea6593e6093`, authenticated against pushed
`022bedf24a9f5fb5080281abd7517341c0da72c7`, returned
`SOURCE_WHEEL / STATUS_BEFORE_TIMEOUT` at `2026-09-28T12:06:19.384887`.
This establishes that the pre-archive tracked-status subprocess exceeded its
30-second deadline in this attempt. It does not retrospectively classify the
older failure or establish why this filesystem check was slow.
[Microsoft's filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
documents slower cross-filesystem work under `/mnt`, which is relevant context
for this checkout, not a measured root cause. The small corrective experiment
allows 120 seconds for each tracked-status check, retaining HEAD's 30-second
limit, exact-commit archive, clean-source rejection, and no replay. No repository
move or change to the model settings or provider workflow is introduced.
Independent review approved the adjustment; 33 source-wheel tests and 150
host/projection tests passed. Skills are synchronized, no locked source or pins
changed; the next fresh live attempt was required to demonstrate completion.

### CPU qualification passed; marker invocation shape failed

Attempt `modal-848bad657119228d331ec45e` authenticated source `a8f102ad` and
advanced through build and CPU qualification into staged submission. The host
reported `marker_readback` and `RUN_SUBMIT_RECONCILE_REQUIRED`. Its exact submit
claim is `4e5ad1b4e6d2fe4bd19c18317a27831e34f11952b343a3897015e4064dc6ef4d`;
there is no retained packaged-call row. Preserve the claim and resources rather
than interpreting that absence as replay authority.

The checked-in read-only marker inspector returned control `MATCH`, artifacts
`UNAVAILABLE`, and model-cache `UNAVAILABLE`. The first marker is present and
readable at inspection time; the other results are inconclusive. A network-free
probe with the pinned Modal 1.5.4 launcher reproduced rejection of
`synchronizer.create_blocking(reader.read_exact)` for a bound async method,
the invocation shape used by the host. The inspector instead wraps a free async
function. This establishes an invocation defect; the retained stage alone does
not independently prove the historical exception. Provider test doubles must
reject that unsupported shape, and the corrected production path needs a real
pinned-dependency, network-free probe before promotion.

The corrected production helper passed that pinned Modal 1.5.4 worker-thread
probe. Root's integrated transport/composition/standalone/inspector gate passed
183/183 tests. A parallel worker suite had 123 passes and one different
standalone simulated-start case (`STAGED_INPUT`) fail before the mocked worker
poll, again projecting `RUN_START_INDETERMINATE`. This recurring test
intermittency remains unresolved and must not be hidden by the clean root run;
no production change was made on an unproven explanation. The bridge fix has
independent static review, both source locks are CURRENT, and skill copies match.
A fresh attempt was required to establish live submission and training.

### Remote admission reached on the free-wrapper commit

Attempt `modal-5dd7b5227af66dc4e0d16863` authenticates pushed source
`794792cde6da7f8c36cb98f5311a83403b7ceef8`. Its retained submit digest is
`e4f7b06282822e1be39b20fc97af6c52aeb94d7bcada048fb83d54dfe1fde5ba`
and packaged call is `fc-01M3MDMESQVC8JRG6N51ZWWF9H`. The ordinary train
command returned `RUN_WORKER_SFT_ADMISSION`; the exact claim-bound read-only
inspector independently returned `WORKER_SFT_ADMISSION`. Thus host markers,
dispatch, Volume binding, and private input/path setup advanced to the generic
SFT admission boundary. No optimizer steps or adapter are evidenced.

A provider-free replay authenticated the retained binding and source, matched
the compiled workload and artifact-policy digests to the execution binding,
and passed `_admit_contracts`. The remaining ordered admission checks inspect
the installed release, hold physical directories, validate prepared bytes,
admit the environment, construct the invocation, and bind physical state. CPU
qualification exercises installed-release inspection in its child but does not
prove the training parent's checks or attempt-specific data/paths. The broad
admission label cannot select a failing subcheck. The next correction adds
closed substage labels without relaxing admission or exposing exception text;
the existing attempt is preserved and cannot be replayed.

The seven admission substage labels passed their injected-boundary tests and
independent review. The fixed five-member packaged-worker manifest was refreshed
for the reviewed source hash without changing its inventory or runtime pins.
An initial broader test run under local CPython 3.10.12 had 47 passes and 20
artifact-evidence failures: targeted instrumentation located the unchanged
archive helper's use of `hashlib.file_digest`, available from Python 3.11.
The full runtime file then passed 67/67 under the pinned CPython 3.11.14 with
the existing pytest harness imported read-only; no packages were installed and
no trainer code was changed to accommodate the older test interpreter. This
explains the local test failures, not the remote admission failure. The final
worker/inspector/host-reader/standalone projection gate passed 267/267 under
the same 3.11 interpreter without flakes. Packaged worker, Modal runtime (98
members), and inference (119 members) checks all report CURRENT; skill trees
match. These gates approved fresh exact-source qualification, not GPU success.

### Installed-runtime boundary isolated on the diagnostic commit

Attempt `modal-4f928f2247331ab8aaaa0f29`, source
`37d7b1f6a26445871373ad7d73f04d53dcd6e55e`, passed CPU qualification and
returned `RUN_WORKER_SFT_ADMISSION_RELEASE` at `2026-09-28T12:57:53.228552`.
The exact claim-bound inspector confirmed `WORKER_SFT_ADMISSION_RELEASE` for
call `fc-01M3MF6HQXRRHXMZ2G6YE6KQNP` and submit digest
`0fe95ea4c651227c6988a9b3d0018636736954e5cb9db6c93f792fe84eb3adc0`.
This identifies the installed-runtime check, not its failing predicate.

The retained release pins CPython 3.12.3 at
`/opt/unsloth-venv/bin/python3`, matching the Qwen packaged image profile.
The operator launcher's CPython 3.11.14 is a distinct runtime, not evidence of
the remote parent interpreter. The CPU child explicitly invokes the pinned
release executable; its parent only checks the pinned wheel/trainer reference.
Training also invokes `_inspect_release` in the Function parent. Commit
`c5257031` intentionally made the generic child qualifier parent-process-agnostic;
retain that property. The next targeted change adds the same runtime inspection
to the authenticated Modal CPU parent and reports finite predicate codes before
another GPU attempt. No interpreter mismatch, inventory mismatch, or other
specific cause is established yet; no pins or validation were relaxed. The
Modal-only parent precheck and finite predicate projections passed 310/310
provider-free tests under the pinned CPython 3.11.14 test harness. Coverage
includes no child invocation or artifact/control commits after parent failure,
all finite host/CLI/inspector projections, and unchanged generic runtime checks.
The five-member packaged closure was reviewed and refreshed; both Modal source
locks remain CURRENT (98 and 119 members) and skill copies are synchronized.
The ensuing live operation was CPU-only qualification, not a GPU training retry.

CPU attempt `cpu-qual-0cda3ea8806cf7eb2788ab4e` on
`b3d1f809963eba8150a1135491009fc6f1f890e4` returned
`CALL_PARENT_RUNTIME_PYTHON_EXECUTABLE` at `2026-09-28T13:19:29.101886`.
The read-only inspector independently matched the same fixed failure on call
`fc-01M3MGECABKF2SJQCDK3KS9BJ1`, claim
`qualify-743bb87bc907af78838822e2272b304917904bb9393a20295dce22f75b88d229`.
Implementation and version checks passed before this literal-path mismatch;
physical interpreter identity and subsequent runtime checks were not reached.
Modal's existing-image documentation and pinned SDK expect `python` on PATH,
but do not establish the actual parent executable or equivalence with `python3`.
The next CPU-only experiment classifies physical equivalence without accepting
the mismatch: same reviewed venv bin and prefix, same regular binary identity
and locked digest, plus Linux running-executable identity. Every result remains
failed with no child or receipt, pending a separately reviewed correction. The
diagnostic-only delta passed 328/328 focused tests, including equivalent aliases,
wrong prefix/digest/running inode, symlink retargeting, and unavailable reads.
The fixed packaged-worker manifest was reviewed and refreshed; Modal runtime
and inference locks remain CURRENT, and canonical skill copies match.

CPU probe `cpu-qual-d52e111cd6d5caefb5e1f821` on
`17815f4d196d9ef731dc5cf3969c8f0bde38aad0` returned
`CALL_PARENT_RUNTIME_PYTHON_EXECUTABLE_EQUIVALENT` at
`2026-09-28T13:31:28.874449`. The exact read-only inspector confirmed the
same category for call `fc-01M3MH490KYMCWTA4F7QRJNSPT`, claim
`qualify-ba46be7c4b79a9e15572722452aa4a4717b101980543f12bb16c8f79333493c5`.
This establishes the strict interpreter-equivalence proof for that CPU parent;
the literal spelling check still rejected it before inventory or child checks.
It does not retroactively prove the earlier GPU parent's physical state.
The resulting correction admits only that proven equivalence in parent checks,
retaining strict child executable admission, exact child invocation, and every
subsequent runtime/wheel/inventory predicate. Fresh CPU qualification must pass
before a new bounded GPU attempt. The parent correction's established five-suite
gate passed 336/336 under the pinned launcher interpreter. An expanded child
suite exposed two pre-existing verification limitations: that launcher's build
lacks `os.memfd_create`, and two synthetic child fixtures supplied `release: {}`
despite the parser's required schema discriminator. System Python 3.10 exercised
the memfd path and isolated five stale-fixture failures (32 passed); independent
review traced all five to that missing schema before the intended callbacks.
Repair only the test records, retain child parser strictness, and rerun that
suite under the existing memfd-capable interpreter. The corrected child-only
suite passed 37/37 under system Python 3.10. No production child/parser or
training pins changed. The packaged-worker manifest and both Modal source
locks are CURRENT; the canonical skill copies matched at promotion.

CPU qualification `cpu-qual-75c4a5d815211fdbb8bc99e2` on
`29faf9d3fa88f7e2f1bad01912d2eddf06003947` returned
`CALL_PARENT_RUNTIME_INSTALLED_RUNTIME` at `2026-09-28T13:47:15.222113`.
The exact read-only inspector confirmed the boundary for
`fc-01M3MJ17ZR1N14RQFZAZ0F1HFD`, claim
`qualify-96b6c97c9a33fd6a19f7a16ba0c5675386e8f544bba87a7c32fa822ebe1ae068`.
Thus parent interpreter equivalence and locked binary digest passed, while the
inner installed-runtime inspection failed before outer measured-record comparisons.
No wheel, provenance, or duplicate-distribution cause is established yet.
The same isolated-child inspection cannot prove the separate Function parent's
metadata/import graph. Add finite inner predicate diagnostics to the existing
inspector and CPU workflow, preserving its wheel/member/closure/inventory checks;
do not remove parent verification or reuse inference-image evidence as a fix.
The diagnostic-only inner-stage change passed 401/401 focused tests under the
pinned launcher test harness. Independent static review confirmed original
predicates and order are preserved. Both changed runtime files remain in the
same reviewed five-member manifest; hashes were refreshed, both Modal source
locks remained CURRENT, and skill copies matched for CPU-only qualification.

CPU attempt `cpu-qual-7773054a7c332b42104f9c7b` on
`c83b17c411d122f0cdd1f8c337bc61c42cdcd56b` returned
`CALL_PARENT_RUNTIME_INSTALLED_INVENTORY_DUPLICATE` at
`2026-09-28T14:01:23.752802`. Exact-call inspection confirmed it for
`fc-01M3MJV27EX2JCPTN5FE7KF8KT`, claim
`qualify-9a3d8aea7cb3049d9da4c643565dccb1aeebc0d3626e9827953cb99e24fc587b`.
All preceding wheel/version/dependency/provenance/member/closure checks passed.
This establishes duplicated normalized names, not yet their physical identity.
The existing inference inspector provides a reusable strict identity pattern:
collapse only repeated stable physical metadata objects with identical normalized
name/version; reject separate or unproven duplicates. Adapt that inside the
training inspector while retaining both raw and unique 4096 bounds and the final
exact inventory digest/count. A fresh CPU candidate must prove the behavior on
the actual parent; no package removal, pin change, or name-only dedup is allowed.
The frozen conditional identity adaptation passed 406/406 provider-free tests
under the pinned CPython 3.11.14 launcher test harness after reviewed five-member
closure hash refresh. Independent review approved a fresh CPU-only qualification.
Both Modal source locks remain CURRENT and canonical skill copies match. No
physical-repeat cause or GPU qualification is inferred from the local tests.

CPU attempt `cpu-qual-a1a525ac11de0a232a9a7e6e` on
`573ad7b7e098772d64974500841e51450f91dc58` still returned
`CALL_PARENT_RUNTIME_INSTALLED_INVENTORY_DUPLICATE` at
`2026-09-28T14:13:54.573947`. Exact-call inspection confirmed it for
`fc-01M3MKHW04GDR3BHGB4XP7YD2P`, claim
`qualify-06a32437b2e7845f443f420830706a3db894fd49ba4ae53601b02b5d689570f6`.
The physical-repeat candidate did not resolve this failure. The next diagnostic
must distinguish duplicate reason, authenticated main/bootstrap/other category,
and inside/outside the authenticated Python library roots in one fixed result.
It remains rejection-only: no raw metadata, paths, package names, versions, or
exception text; no parent inventory scope change and no GPU qualification.
The [Modal SDK release notes](https://modal.com/docs/sdk/py/releases) describe
changes to client-dependency inclusion with the 2025.06 image builder; the
[existing-image guide](https://modal.com/docs/guide/existing-images) also
distinguishes Function-compatible Python from arbitrary image use. These are
hypothesis context, not evidence of this attempt's duplicate origin. Repository
review establishes that the build capture and actual trainer use isolated Python,
whereas the Function parent performs privileged model preparation and storage
coordination. Any eventual inventory-scope correction must account for that
privileged parent's imports rather than simply removing its checks.
The diagnostic-only first-pair classifier passed 660/660 focused tests under
the pinned CPython 3.11.14 harness after reviewed fixed five-member closure
refresh. Both separate Modal locks remain CURRENT and skill copies match.
Independent static review found no admission or inventory-equality change; actual
pinned Modal 1.5.4 serialization of all 78 parent-stage results measured at most
224 bytes, below the unchanged 512-byte diagnostic bound (network-free check).

CPU attempt `cpu-qual-24769c0aab0773e729a8cde9` on
`56cb974ac8c906bf5c2d9bd46c8a779bde47828b` returned
`CALL_PARENT_RUNTIME_INSTALLED_INVENTORY_DUPLICATE_VERSION_MISMATCH_OTHER_CROSS_ROOT`
at `2026-09-28T14:28:55.318997`. Exact read-only inspection confirmed the result
for call `fc-01M3MMDDEPD1YSJK3MA9P55MPZ`, claim
`qualify-a508561d9096e5e1ceb21bd9cd6692a7ce2aed3c774631c8db5f7def362f33c8`.
The first offending pair is version-mismatched and spans the authenticated
library-root boundary; neither name belongs to the main/bootstrap wheel list.
The result does not identify the package or prove who supplied the outside copy.
No GPU ran. Review a proportional parent/child inventory correction before edits;
retain parent source/bootstrap/closure checks and strict isolated child pins.
The selected next experiment is diagnostic-only: on that exact measured stage,
compare a bounded direct stdlib metadata scan of authenticated, stable library
roots with the release digest/count and run the existing strict isolated child
qualifier. Every included metadata object must be proven in-root; uncertainty
is UNPROVEN, never MATCH. Return one fixed combined failed result regardless of
both outcomes, before signing or committing any qualification receipt. This
measures installed directory metadata and isolated execution, not parent-loaded
module provenance; normal admission and GPU authority remain unchanged.
The frozen probe passed 688/688 focused tests under pinned CPython 3.11.14
after reviewed worker-closure hash refresh. Both other Modal source locks remain
CURRENT and skill copies match. Actual Modal 1.5.4 serialization of all 84 fixed
parent failure documents remained at most 224 bytes. Independent source review
confirmed the exact-stage trigger, unchanged normal admission, and no receipt
signing or output commit for all six probe outcomes.

CPU attempt `cpu-qual-2d0ee360c4466b02eb1a80c2` on
`0d7ff127b6033d8080fd5ffc1020e79c02bb3428` returned
`CALL_PARENT_RUNTIME_AMBIENT_ROOT_MATCH_CHILD_PASS` at
`2026-09-28T14:45:47.737633`. Exact read-only inspection confirmed the result
for call `fc-01M3MNCARNZK50RRFPB67VHSBA`, claim
`qualify-bb9290cb432c9ae58cfa801dcada1cbf717c6f916dba665a0b1fd12e35ec26c4`.
Both the complete reviewed-root metadata inventory and the strict isolated child
match the saved release. The diagnostic intentionally issued no success receipt.

Independent review recommends an explicit policy decision before changing parent
admission. A parent-only reviewed-root inventory check would preserve Python,
main/bootstrap wheel bytes, provenance, installed members, closure and trainer
assets, plus the exact release inventory under proved image roots. The child
would retain full strict ambient inventory in isolated mode. However, the parent
would stop blanket rejection of outside-root metadata overlays, which could
supply its model-preparation or provider dependencies. Neither metadata inventory
nor current installed-wheel checks attest already-loaded module origins. The
outside copy is not proven unrelated or provider-owned. Obtain explicit operator
acceptance of that parent trust trade-off, or review a narrow critical-import
safeguard, before implementing the scope change. No corrective admission code
has been written; no cloud job is active and no GPU/LoRA success is claimed.

Current disk check: F: has approximately 324 GiB free; WSL root approximately
231 GiB available. Disk pressure is not the current blocker.
