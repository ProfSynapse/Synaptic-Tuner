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
231 GiB available. Disk pressure is no longer the current blocker.

Operator approval (2026-09-28): Joseph explicitly approved the disclosed
parent-only inventory scope correction, regression tests, CPU qualification and
L40S smoke. The accepted policy permits extra outside-root dependencies in the
privileged parent; it does not attest loaded-module origins. Implement explicit
reviewed-root inventory selection at `_inspect_parent_release` only, retain
full ambient selection by default and in the isolated child, and preserve all
other Python/wheel/bootstrap/provenance/member/closure/trainer checks and exact
release digest/count equality. Unproven roots and in-root drift remain failures.
The implementation and independent review are delegated separately to Sol;
the orchestrator owns source commitments, live runs, policy docs and journal updates.
The approved correction passed 698/698 supported focused tests under the pinned
CPython 3.11.14 launcher harness, plus 64/64 generic worker and isolated-child
tests under the existing WSL Python 3.10 interpreter with memfd support. The
expanded pinned-launcher run was 721 passed / 4 failed solely at its known
missing `os.memfd_create`; no production workaround or interpreter pin change
was made. Independent review approved the frozen change. The fixed five-member
closure was hash-refreshed; both other Modal locks remain CURRENT and canonical
skills match their synced copies. The resulting pushed source is
`07ae159a77b65b3a1cf3a24d8f420efe0e82af66`.

CPU attempt `cpu-qual-8fc201ecf8edfae5a34350b5` succeeded at
`2026-09-28T15:15:52.705483`. This is an actual signed CPU qualification,
not the earlier diagnostic-only root/child probe. The runtime release digest is
`d2aaa0adf764129dc06d94c48de7995aad814c88c334f3070856947d80e7e4b1`;
qualification output SHA-256 is
`3d426c9ee3c4925d4e88905911531c417e016a18ab89d0a672229387cc2399fc`.
It does not establish GPU execution or saved adapters.

Fresh L40S attempt `modal-13bf59329010166ee092513c` stopped at
`RUN_WORKER_SFT_PREPARATION` at `2026-09-28T15:21:10.516836`.
The authenticated exact-call inspector confirmed `WORKER_SFT_PREPARATION`
for call `fc-01M3MQBKGA17WZ65YYW499MRQ0`, submit claim
`5d06f4be0793807f2a51af72f7729b8ab0b97956fef2c441a7c6fe80cb4828a9`.
This passed the prior runtime boundary but does not establish successful model
download, cache commit, optimizer steps, or durable LoRA artifacts. The attempt
is consumed and will not be replayed.

Independent read-only diagnosis found that this closed stage combines Hub API
admission, pinned metadata, private download/verification, persistent cache
publication, cache commit, exact snapshot path, and initial snapshot inventory.
Existing retained records and inspectors do not distinguish these operations;
the model preparer deliberately suppresses raw SDK output and exceptions.
The next targeted experiment adds only finite non-secret substages across these
boundaries, with regression tests and independent review before a fresh run.
No underlying cause is inferred from the broad failure phase alone.

The preparation diagnostic patch preserves the model-preparation predicates,
cache/artifact/control commit order, credential handling, and one-shot claims.
Shared model preparation owns its finite exception locally; only the packaged
Modal wrapper translates it, avoiding a new training dependency in inference.
Independent source review approved the change. Worker, host and exact-call
inspector allowlists agree on all 49 fixed stages; actual pinned Modal 1.5.4
serialization measured at most 277 bytes against the unchanged 512-byte bound.
Existing inventories remain fixed: worker closure five members, Modal runtime
98 members, inference 119 members. Only reviewed source hashes were refreshed.
The root regression gate passed 980/980 on pinned CPython 3.11.14; generic worker
and isolated-child tests passed 64/64 on the existing memfd-capable WSL Python.
The additional wrapper-translation gate passed 22/22 on pinned CPython 3.11.14,
including exact model-stage translation, subclass fallback, cache-commit failure,
no artifact/control commit after preparation failure, and unchanged success
ordering. All source commitments remain CURRENT. Independent final review
approved a diagnostic commit and fresh CPU/GPU attempt, not a cause or training
success. The journal and roadmap retain the failed attempt without replay.

Diagnostic source `c0ac2f96c6700913be482540e31db1b79a093061` was pushed.
Fresh CPU qualification `cpu-qual-a74cd731e39467cada8d3fdd` passed at
`2026-09-28T15:49:02.503499`, runtime release
`78556634e53bca8d42307593b881e9d8f0ae7dca36da5f24edcc02c48d6ef039`,
qualification output `294911a2b701ce16bf147affe9a0bb0da512824fa94263b9285d7703537ceec7`.
The fresh L40S attempt `modal-206e29eaaf07fd37b054a48a` stopped at
`RUN_WORKER_SFT_PREPARATION_MODEL_PERSISTENT_PUBLICATION` at
`2026-09-28T15:55:51.704447`. Exact-call inspection confirmed the corresponding
`WORKER_` result for call `fc-01M3MSAE0V6QADGKKABEPZXFJ1`, claim
`31994bb43760cf64089bf509ffd3305cd987a45af5a7cdc8196e89511fb7efbb`.
Hub metadata, private download and initial file verification reached publication;
the explicit cache commit and trainer were not reached. Modal background commits
mean failure before explicit commit does not imply no persistent effects.

Publication still combines create-only directory claims and exact file copies.
Existing model-preparer and entrypoint tests substitute a simple bound-cache
fake rather than exercising the actual descriptor-bound publication operations.
The real Linux binding/model-preparer integration passed under umask 022 but
failed at persistent publication under umask 002. Default-created repository
and SDK directories can be 0775; the existing private-chain guard rejects their
group-write bit. This proves a local compatibility bug, not Modal's remote umask
or the cause of the retained remote failure.

Independent review approved explicitly creating the private repository and
validated member-parent directories at 0700 before SDK writes. Create each parent
depth-first; `mkdir(parents=True, mode=0o700)` does not constrain intermediate
parents. Do not change process-global umask or relax private-chain/Volume guards.
The corrected real-binding test passes under both umasks, including nested
members. To keep the workflow proportional, defer the broader binding diagnostic
draft and promote only this small permission correction with its regressions.
The existing closed `PERSISTENT_PUBLICATION` label is sufficient for the next
experiment; finer diagnostics remain available if the failure persists. No path,
raw errno, SDK text, credential, or arbitrary exception may escape. Retain every
validation, immutable inventory, commit order, and consumed attempt. The fresh
remote experiment is still required for attribution.
The root regression gate passed 666/666 under pinned CPython 3.11.14, including
real publication, mounted I/O, inference model/worker integration, packaged SFT,
entrypoint, qualification, and host diagnostic checks. The generic worker closure
remains unchanged and CURRENT; both fixed Modal source inventories received only
reviewed model-preparer hashes. Skill copies are synchronized.

Permission-fix source `99a04b79f163fab21140632c01940823526cd1a4` was pushed.
The supplemental inference-preparation suite passed 79/79. CPU bootstrap
`cpu-qual-93eeaf4cbc91c35ad7d0ffc7` was interrupted with only retained claim
`build-263b4fca6b72ccbe0725b1c4ff57ccc114047ff8e925dd90ddfe41a0b014a2fd`
and no catalog rows. Its provider effects remain uncertain; it was preserved,
not replayed. Fresh CPU qualification `cpu-qual-7ee8e2ac563bf0e57809c7d7`
passed at `2026-09-28T16:41:34.975804`, runtime release
`28c3537c22095e7006baff175e696fa02203d4e5f5c0f06883df9f2ec024a9c3`,
qualification output `5038bb480b120fd38f82d24226327748e92291e0e20fb89bbd50fc09eabe377c`.

Fresh L40S attempt `modal-18ee4a76b58d8645178953b6` on this source still failed
at `RUN_WORKER_SFT_PREPARATION_MODEL_PERSISTENT_PUBLICATION` at
`2026-09-28T16:49:14.432862`. Exact read-only inspection confirmed the stage
for call `fc-01M3MWCR2GWZFMB43QSE4S4NM4`, claim
`ddc4da94b848a6b0e55918d80c7658fc6eace20adc1aa3b8c54e26031b6361ea`.
The permission correction fixed a demonstrated local bug, but did not resolve
this remote failure; do not attribute the old or new failure to umask.
No optimizer steps, explicit model-cache commit, or durable LoRA are proved.
Preserve this consumed attempt and narrow the publication operation/predicate
using the deferred finite diagnostic design before selecting another fix.

On 2026-09-29, the finite publication diagnostics completed independent static
review. The implementation preserves rejection behavior and exposes only closed
source-chain, directory-claim and copy-operation labels. The agent gate reported
361 passes and one skipped serializer check; the orchestrator then ran that check
with the existing CPython 3.11.14 / Modal 1.5.4 launcher, and both serialization
size and all-reader allowlist parity passed. No dependency installation was needed.
The handoff had stalled after agents completed, not during a running cloud job.

Two independent source traces confirmed that normal standalone `train` invokes
the signed CPU qualification on its own freshly deployed runtime before public
training start. Tests cover both bootstrap/CPU/stage/spawn ordering and CPU failure
blocking GPU start. A separate fresh `--qualify` checks a different deployment
and its receipt is not reused. This attempt will use the normal train path's
mandatory gate, not a redundant standalone qualification. This does not bypass
CPU proof or claim that CPU proof establishes GPU/model success. The final frozen
877-test combined gate passed in 100.08 seconds, including the pinned serializer.
The binding/model/mounted-I/O suites passed 98 tests; the corrected entrypoint
suite passed 23; generic worker/isolated-child regressions passed 64. A preceding
combined run had one suppressed fake-start failure that passed alone, in the
157-test standalone suite, and in the final combined rerun; its cause remains
unproven rather than assigned to the diagnostic mapping. All three source
commitment checks are CURRENT and skill copies match. No runtime/model pins or
locked inventories changed. The next fresh attempt is a diagnostic smoke, not
evidence that the remote publication problem is fixed.

Source `1b43fdd0d651593151e04cfa454bb9d86cd103e9` was pushed. Normal train
passed its integrated CPU gate, then attempt `modal-c7765719c5d8504dd70f9e91`
failed at `RUN_WORKER_SFT_PREPARATION_MODEL_PERSISTENT_PUBLICATION_SOURCE_CHAIN_TMP`
at `2026-09-29T09:53:19.619620`. Exact claim
`886855784118036b02e1ae409ba628b607ace64bb333488cd3c885a2e7990c00`
and call `fc-01M3PPZTNT9WB7XTVKWANTPQAD` were confirmed through the read-only
claim-bound inspector. This proves rejection at the private source ancestry's
`/tmp` check, not a Volume copy or fsync failure. That check combines directory
type, owner 0 and exact 01777 permissions. Preserve the consumed attempt; split
only this remaining composite diagnostic before selecting a corrective policy.
The remote ownership/permission category and optimizer steps remain unmeasured.
The next diagnostic retains every rejection and separates `TMP_OWNER` from
`TMP_MODE_NONWRITABLE` and `TMP_MODE_WRITABLE`. The latter two describe only
group/other write bits (`mode & 0022`), not the owner's access or sticky-bit
state. This probe still admits only root-owned 01777 `/tmp`. Independent review
approved the diagnostic-only change; 837 focused regressions and four exact TMP
public-CLI projection cases passed; all fixed source commitments were CURRENT.

Source `63fae284647943a2a733f90e78c0e6fb03705242` was pushed. Attempt
`modal-5ea9662614bfb8cfc464f9a6` failed at `SOURCE_CHAIN_TMP_MODE_WRITABLE`
at `2026-09-29T10:06:08.387925`. Exact claim
`0fa0ceb24ee9d3d97db6473fb219202e1cf9713b02a03c26bcd6cc097a383e1b`
and call `fc-01M3PQQ8SSRW6Q9WS3F0Q6XK33` were independently read back by
the claim-bound inspector. `/tmp` was therefore a root-owned directory with
group/other write bits and a mode other than 01777; exact mode/sticky state
remains unknown. Do not relax this guard or chmod the shared parent. Relocate
per-call 0700 staging to a parent that satisfies the existing private-chain
checks, preserving every model, descriptor, publication and no-replay check.

Independent review selected the existing container root as the scratch parent:
the exact failed call already passed its retained root-descriptor trust check.
Unlike `/root`, it introduces no unmeasured intermediate ancestor. Only the
fixed parent selection changes from `/tmp` to `/`; `TemporaryDirectory` still
creates an unpredictable 0700 child and cleans up only that child. There is no
fallback, shared-parent chmod or private-chain relaxation. Rootfs writability is
consistent with [Modal's local filesystem example](https://modal.com/docs/guide/volumes),
but an exact-image run remains necessary to prove preparation and training.

The selected-root real binding copy passed under WSL root (one test). The pinned
unprivileged launcher passed 370 focused regressions with the root-only test
skipped; that skip is covered by the separate privileged execution, not counted
as a pass. Independent review approved the minimal change. All three fixed
source-commitment checks remained CURRENT without hash, inventory or pin changes.
That gate authorized one fresh normal train invocation with integrated CPU proof.

Pushed source `d93b6c8b` advanced attempt `modal-1fbd16d341752739b3a9088d`
through model preparation and returned `WORKER_SFT_TRAINER` at
`2026-09-29T10:27:21.456653`. Claim
`2c638df741a8b36dace60f592094b3e987528f06718f2deaeddd25440b1d50e5`
and call `fc-01M3PRMKX8VZENXT8KXH323BG3` were confirmed by the exact
read-only inspector. No optimizer-step or adapter-success claim follows.

Source tracing found that the bound v2 worker's stdout/stderr files live under
the per-call private temporary directory and are removed on exit. Nonzero child
exit and runner-start exceptions collapse into the same TRAINER stage; failed
runs do not publish those files. Public logs are not qualified for this path.
The next change must preserve a bounded closed diagnostic across this boundary,
not guess OOM or change training configuration. This attempt remains consumed.

Independent review selected reserved nonzero child exit codes instead of log
parsing, a sidecar or another monitoring service. Six explicit child boundaries
(transport, release/contracts, input/environment, import guard, trainer entry,
postcheck) use five fixed categories; trainer entry additionally distinguishes
five exact built-in error types. The host maps only those 35 reserved codes to
closed stages. Unknown exits and signals remain generic, never inferred OOM.
The existing bounded result, exact-claim inspector and public reader carry the
labels. Messages, paths, traceback text, locals and log bytes stay private.
No success, credential, publication, training-config or replay policy changed.

Verification passed 685 worker/reader/inspector/projection checks, 222 execution
and entrypoint checks, 104 isolated-child checks on memfd-capable Linux Python
3.10, and 27 generic-worker checks. The pinned 3.11 launcher cannot run the
eight existing native memfd child cases; those passed in the 3.10 child suite.
All fixed source commitments are CURRENT after refreshing only the two changed
members of the existing five-file worker closure. Independent static review
approved the finite diagnostics and CPU-dispatch isolation.

A recurring provider-free CLI projection test stopped before the fake provider
call with `RUN_START_INDETERMINATE`, first at COMPLETION and then at a different
parametrized label. Retained test state contains marker material and a stage
receipt but no call catalog. This predates the child diagnostic change; tracing
narrows it to the pre-spawn transport boundary but has not established a cause.
Two instrumented full 117-case runs passed. Keep this risk separate and open;
neither timing nor random marker content is asserted as the cause. Independent
review permits one fresh diagnostic smoke only after a clean uninstrumented
117-case gate also passes, CURRENT locks and diff checks; another gate failure
holds promotion. This does not authorize replay or claim the flake was repaired.

The clean uninstrumented gate passed all 117 cases in 113.67 seconds. Temporary
test tracing was removed with no remaining diff in that test file. Skill trees
are synchronized and diff checks are clean. The unresolved pre-submit risk is
tracked in the private project as `T-22a498d3`; conditional review clears one
fresh normal-train diagnostic attempt, with mandatory integrated CPU proof.

Pushed source `bc22dfdb` passed its integrated CPU qualification. Attempt
`modal-fe9e13ba3f821d9e3ce35364` returned
`WORKER_SFT_TRAINER_CHILD_EXEC_RUNTIME` at `2026-09-29T11:19:42.776996`.
Exact claim `5d9000a79ced83f40377217e386a18611ea5619e57bd79ab5734313f744d7972`
and call `fc-01M3PVN2SXGWXAYSQY71GBW24D` were confirmed by the claim-bound
inspector. This identifies an exact built-in RuntimeError during trainer exec,
not its operation, message, or cause; it is not evidence of an OOM. No optimizer
step or saved adapter has been proved for this exact path. Preserve the attempt.

The next diagnostic refines only that RuntimeError through a child-owned finite
in-memory operation marker. It distinguishes model snapshot validation, library
load, model and tokenizer source checks, loss guard, dataset preparation, LoRA,
trainer setup, training and saving. It introduces no diagnostic file, raw log,
traceback, result field, or guard fallback. Ordinary trainer calls without the
hook retain their behavior. The suspected processor/tokenizer source mismatch
and memory-efficient-loss guard remain hypotheses pending measurement.

The frozen diagnostic includes 17 operation/import milestones (reserved exits
80–96); the prior 35 diagnostics remain unchanged. Independent static review
approved child-only callback injection, exact-RuntimeError specialization,
unknown-marker fallback, and unchanged source guards. Focused verification
passed 137 child/loader checks, 205 execution checks, 322 reader/inspector checks
including the pinned 512-byte serializer bound, and 16 trainer-source checks.
Hash-only refreshes updated the two trainer members in the fixed 66-member
offline closure, two diagnostic members in the five-member packaged closure,
and the offline-manifest hash in the 98-member bootstrap lock. Inference's
119-member commitment stayed unchanged. No model or dependency pins changed.

A broader isolated-runtime test cannot run in the current test interpreters:
the pinned launcher lacks isolated jsonschema, and system Python lacks isolated
referencing. Two older source-string tests also assert names already absent at
HEAD; they are not made green by changing the trainer. These limits are separate
from the focused passing suites and are not reported as passes.

Paid promotion remains held after the local public-CLI stage test again returned
RUN_START_INDETERMINATE. Test-only tracing captured a FoundationError with
authority_invalid before provider spawn. The exact authority predicate is still
being investigated at that checkpoint; a passing rerun was not treated as repair.

The subsequent bounded predicate probe captured the cause of this local test
failure at `foundation_v2/broker.py:127`: grant execution time was one second
earlier than grant activation. All command-content, authority, epoch, revocation
and HMAC predicates passed. Only `not_before <= now < expires` failed, with
`now - not_before = -1` and `expires - now = 901`. The retained synthetic fixture
is `/tmp/modal-grant-predicate-1-20260929/test_exact_worker_failure_stag7`.
This establishes local wall-clock rollback, not a Modal/provider defect or the
cause of the separate remote trainer RuntimeError. The verifier correctly failed
closed; do not clamp time, backdate grants, relax expiry, or replay an attempt.

Read-only environment checks found WSL host timesync enabled (`timesync_implicit=Y`),
guest systemd-timesyncd active/enabled, an NTP offset of -780.528ms, and the Windows
Time service stopped. These observations suggest competing time sources but do
not prove the mechanism. Similar backward steps are reported in the
[Microsoft WSL tracker](https://github.com/microsoft/WSL/issues/11790) and
[timesyncd discussion](https://github.com/microsoft/WSL/discussions/11548).
Independent review favors a temporary, reversible guest-NTP-stop experiment,
with clock observation and the clean provider-free gate, rather than changing
authority semantics. Explicit user approval is requested because stopping that
service affects the whole WSL instance. Time-service state remains unchanged.

Temporary test tracing was removed. The independently reviewed deterministic
broker regression issues a valid stage grant at epoch 101, executes at 100, and
asserts AUTHORITY_INVALID with no resolver/provider call or effect record.
All 25 tests in the bounded-remediation file passed under the pinned launcher.
This protects the correct fail-closed behavior; it does not repair the WSL clock.

The user explicitly set the time-service experiment and authorization redesign
aside and directed continuation of safe training. Independent review grants
conditional GO for one fresh normal TrainingAPI diagnostic attempt on the
reviewed milestone build, after a clean uninstrumented stage-projection gate,
CURRENT commitments and an exact pushed source. The known clock regression
remains a fail-closed environmental risk, not a reason to bypass checks. Normal
train retains its integrated signed CPU qualification before GPU dispatch.
No OS service, grant semantics, model pin or replay policy changes are authorized
by this decision. Any recurring host failure requires exact-state diagnosis;
no consumed or uncertain attempt may be submitted again.

The clean live-clock projection matrix stopped after 118 passes with another
pre-submit indeterminate result; that fresh failure is not assigned a cause from
its stage alone. To isolate what this matrix actually tests, its single
parametrized case now supplies a fixed paired UTC string/epoch through the
existing clock port. Independent review approved this test-only change. It does
not enter the real launcher, alter authority, suppress failures, or remove the
separate deterministic rollback-negative regression. The environmental risk
remains recorded while the training objective proceeds through ordinary guards.

The isolated projection matrix passed all 134 cases. Pushed source `b65f4829`
then completed integrated CPU qualification and submitted fresh GPU attempt
`modal-62f366306b2d17940efedebe`. At `2026-09-29T12:41:11.194541`, the exact
claim-bound inspector and public CLI agreed on
`WORKER_SFT_TRAINER_CHILD_EXEC_RUNTIME_TOKENIZER_SOURCE` for claim
`97a23f6c3849e5d22389b816c511be26df64f3cb336bb9418edfb86a8c440d04`,
call `fc-01M3Q0BJPN4RQBME12FZM3MV81`. The model library load and model-source
validation returned successfully; tokenizer-source validation failed. This is
not optimizer-step, adapter-save, or OOM evidence. Preserve the consumed attempt.

The existing loader accepts a processor wrapper elsewhere for vocabulary-size
reporting but its protected source check only reads top-level `name_or_path`.
Transformers documents processors as compositions containing a text tokenizer.
The reviewed narrow candidate validates the nested tokenizer against the same
exact snapshot, checks any present wrapper source for consistency, and preserves
the original returned object. The remote object shape remains unmeasured; a
fresh live run must test the candidate. No dependency/model pins, offline flags,
authorization semantics, or OS time services are changed by this candidate.

The narrow implementation passed all 21 focused loader tests under the pinned
CPython 3.11.14 launcher, including nested-source acceptance, processor-object
preservation, conflicting/malformed source rejection and property-error failure.
The nine offline/bootstrap closure tests passed with their own fixture scope;
the isolated launcher does not supply the unrelated root ML fixtures' NumPy.
All four source commitments are CURRENT and skill trees remain synchronized.
Only the model-loader hash in the fixed 66-member offline closure and that
manifest's hash in the fixed 98-member bootstrap lock changed; no inventory,
dependency pin, model revision or training parameter changed. The next proof
was one fresh ordinary TrainingAPI run with its integrated CPU qualification.

Pushed source `25b07cff9e72c8b0780bca63a66bbfa1a17ccfd7` launched attempt
`modal-38a213ad45839fdb6e9d62fb`, passed integrated CPU qualification, and
submitted claim `8e52e8b5332ef0495b29ff9ae3644e987eb85716620f4182f851431ecd5b29f9`
as call `fc-01M3Q1RP38V7Z0TTWX27ZRZ254`. Both exact-call inspector and public
CLI reported `WORKER_SFT_EVIDENCE` at `2026-09-29T13:07:44.798030`.
This stage does not identify a cause or prove successful optimizer steps:
post-child snapshot/directory revalidation can fail before the core examines
the returned exit status, and successful child output must still pass the
projection, lineage, directory and metric checks. Preserve this consumed attempt.
Read-only diagnosis separates packaged post-child checks from core evidence
validation; a provider-free actual-producer projection comparison precedes any
new paid probe. No guard is bypassed; no successful artifact is claimed.

The provider-free comparison exercised the actual AST-isolated trainer
projection producer and the packaged messages fixture with the production
sealed-dataset helper. Its type-sensitive expected/actual comparison matched.
This rules out a structural mismatch for that fixture, not the live attempt.
The next diagnostic adds eight fixed labels at existing evidence checks:
private-copy identity, held directories, output binding, output inventory,
dataset binding, projection binding, output directory shape, and metrics.
It adds no checks, I/O, result fields, resources, or raw diagnostic content.
An independent reviewer caught and corrected a stale specific label after
successful inventory; the generic EVIDENCE fallback remains for unclassified
later errors. Nonzero child results can still be masked by a failing recheck,
so no label alone establishes training success.

Verification passed 215 packaged-execution tests, four focused core evidence
tests, 338 host/inspector tests (including bounded serializer/parity checks),
nine fixed-closure tests, and 142 public CLI diagnostic projections. All four
commitments are CURRENT; only hashes within existing inventories changed.
The isolated core tests require pinned CPython 3.11.14; native 3.10 rejected
its version contract before evidence, not a diagnostic assertion failure.
The independent Modal-worker allowed-stage inventory was updated by exactly
the eight reviewed labels; its full 130-test file passed. Skill trees remain
synchronized.

Pushed diagnostic source `779e4ddc276bad562421b27ec7122725c530b94d` passed
integrated CPU qualification for `modal-595c5520451e28297b0a8503`. Submit claim
`69c1c22dc5a173488d72238e1e1eabbe7d61e37fcc18e172270c2796c802cc97` produced
call `fc-01M3Q3ZD5DKPY2S2PXSY7350YJ`. At `2026-09-29T13:46:34.960528`,
public CLI and exact-call inspector still agreed on generic `WORKER_SFT_EVIDENCE`.
The generic stage also spans subsequent core artifact assembly; it cannot prove
the failure occurred at an evidence predicate or that training succeeded.

A provider-free experiment added only `preprocessor_config.json` and
`video_preprocessor_config.json` to the existing otherwise-valid runner output.
Core artifact selection rejected it with its fixed unsupported-file error and
no diagnostic code, reproducing the generic wrapper stage. Both filenames occur
in the exact pinned Qwen snapshot. The live emitted inventory remains unobserved;
this is a reproducible producer/contract incompatibility, not yet a live cause
claim. The suspected PEFT base-ref mismatch is already corrected by the existing
post-save stamp and is not promoted as the cause.

The trainer inventory pins Transformers 5.17.0; earlier 4.57.1 processor docs
only supported a general API explanation, not the live trainer version. Its
Trainer save path does not save a processor with this repo's current kwargs,
while the explicit ProcessorMixin save emits component configuration. The
reviewed candidate keeps the processor for preprocessing/training but saves its
text tokenizer for runtime-v1's text-only archive. It must serialize the actual
wrapper chat template while restoring in-memory tokenizer state afterward.
No artifact allowlist broadening or output-file deletion is part of this fix.

The frozen save helper passed 21 focused source/behavior tests on pinned
CPython 3.11.14. Tests exercise direct-tokenizer behavior, differing/missing
inner templates, success/failure restoration, no processor save call, strict
model/tokenizer archive acceptance for text-only output and rejection of both
modality sidecars. Independent review found no issues. The fixed offline
66-member manifest refresh changes only `train_sft.py`; the bootstrap lock
refresh changes only the offline-manifest hash. All four commitments are CURRENT
and skill trees are synchronized. The live candidate still requires one fresh
ordinary run; prior consumed attempts are not reused.
