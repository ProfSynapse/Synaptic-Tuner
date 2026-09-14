# Modal inference image qualification

2026-09-14 implementation record. This is engine-only maintenance work; EHR,
main, and existing remote branches are unchanged by these local edits.

The CPU capture command now offers an explicit private-directory check. It
measures effective-user ownership, exact 0700 permissions, canonical distinct
directories, and actual temporary write/fsync/read access at the four fixed
chat paths. Its temporary probe files are closed and removed on return. It is
an immutable-image check, not a hostile-Volume traversal primitive. Exact
stopped-Sandbox readback validates the same opt-in report fields.

The next package check uses the shared production verifier with an independently
selected runtime-lock digest. It does not create a fake authenticated serving
configuration. The serving entrypoint still verifies its real configuration and
uses the same source, closure, physical interpreter, and complete distribution
checks. A successful CPU package check remains short of live model/chat
qualification; the outer capture report deliberately stays candidate-only.

Initial local measurement: the directory inspector/capture selection passed
121 tests in 0.48 seconds under CPython 3.12.9 and pytest 9.0.2 with automatic
plugins disabled. Independent review found no defects in that directory-only
diff. After adding package-report transport validation, the same selection
passed 128 tests in 0.48 seconds. These counts overlap and are not additive.
No new live image or GPU result is claimed by these tests.

The restored formatter environment lacks NumPy, so a test invocation there
failed at root conftest import before collection. Tests were then run from the
execution checkout with the existing dependency-complete Python environment;
the main Python/Modal installation was not changed.

Consumer audit also confirmed that the published source fixture is not a live
host factory. The current coordinator descriptor disables observation and
artifact streaming, which gates public run verification and hence chat source
binding. Native authenticated metadata readers exist and can verify the exact
five-artifact inventory without streaming model bodies through the operator.
Live reader qualification and real consumer composition remain separate work;
this image check does not enable those capabilities or substitute test evidence.

The three generated resources are now present. The accepted evidence hash is
`0e744c218c2e752e898751714c00a2508811808470f02034e450024824d88c4e`;
the runtime manifest hash is
`c9258e079d32e9623567abefe9ea1c33ba7696ab97773dd8de04d719f1f39ec8`.
The dependency provenance binds the exact additive bytes with hash
`8dd83eedabeb60c6777aeeecf6596a91857236b68343f004f9302c7ccf191248`
and 230 measured distributions. The source closure remains 118 members and
the runtime inventory is those members plus the two neighboring resources.
Both inference (118) and training (97) maintainers report CURRENT.

Independent integrated review passed 198 focused tests in 2.40 seconds and
cleared this source for a bounded CPU diagnostic only. Review caught and fixed
an option-consistency gap: exact readback and the inspector CLI now require
the private-directory check when packaged-runtime verification is selected.
The production verifier intentionally still rejects all duplicate distribution
enumeration, even when the metadata-only inspector could prove a physical alias.
The isolated final image must pass that stricter check; inspection success alone
does not establish it. No runtime check was weakened to accommodate a candidate.

The broader initializer/runtime/inspector/capture/private-directory/bootstrap
selection passed 246 tests in 122.27 seconds. The initial exact-source wheel
from `c5fdffa1ec5de9e9185658e4f84665d88c5faddb` then failed the local packaged
inventory check before any cloud allocation: Windows Git archive converted
`inference-dependencies.lock` to CRLF because it lacked an explicit LF
attribute. The runtime hash correctly rejected it. The additive installer lock
had the same export risk. Both exact paths now declare `text eol=lf`; source
and runtime pins were not changed. The rejected wheel was not deployed or used
as qualification evidence. A fresh archive/wheel must pass byte comparison.

The corrected `e03cdd78b8ca8826f3f7c3eb3ec6949001d468db` wheel is 2,129,612
bytes with SHA-256
`7636d88f6f219bbbea1f1f051dddb3df04d2e0d583e0ae73d83c485720515f2c`.
All 120 locked members passed; 729 packaged files matched the completed
2,446-file Git archive byte-for-byte. The new LF regression passed in a
21-test initializer selection (0.76 seconds).

The first launch was blocked before process creation by the approval layer.
After explicit user confirmation, the CPU capture ran and failed with no
Sandbox ID, unknown ownership, and unconfirmed cleanup. Its original result
is retained in `evidence/modal-inference-e03cdd7-failed.json`; it is not rewritten
as proof of no allocation. Bounded, read-only logs for only the selected app in
`synaptic-smoke-v1` returned `PRIVATE_DIRECTORY_INVALID` through an allowlisted
output filter. This is app/time-window evidence, not an exact Sandbox receipt.

The image helper had required an existing workspace parent. The correction
creates only that immediate parent after a canonical ancestor check, retaining
symlink/collision rejection. The capture now uses the installed SDK's public
`Image.build(app)` before `Sandbox.create`; the selected initialized app supplies
the explicit client/environment. A build failure or incomplete build is now
distinguished before any Sandbox create attempt. The original attempt's cleanup
uncertainty remains recorded. The four-file preparation/capture regression
passed 101 tests in 0.49 seconds; the late-create interrupt injection was moved
to the third operation (lookup, build, create), preserving its cleanup assertion.

## Successful concrete CPU qualification

The corrected source `d6e29ed03e026bd0653967c109dffe69625e2b46` built wheel
SHA-256 `d139289e777c6921491660a8584b5a5640ef1fdca7f4e97e936282d93579b1c2`
(2,129,612 bytes). All 120 locked members and 729 shared archive/package files
passed byte comparison against the completed 2,447-file Git archive.

The bounded CPU capture succeeded in Sandbox `sb-0zoIiXBJs78ViKq11C54xf`, using
hydrated provider Image `im-rwqrQYtujjHal3RpL3RYW5`. It verified the actual
CPython 3.12.13 executable and exact 230-distribution environment with the
production packaged-runtime core. Its private-directory probe verified UID 0,
mode 0700 and actual read/write access at all four fixed paths. The unchanged
strict production duplicate-distribution check passed; no deduplication bypass
or guessed dependency pins were needed.

Runtime manifest digest:
`c9258e079d32e9623567abefe9ea1c33ba7696ab97773dd8de04d719f1f39ec8`.
Source-inventory digest:
`53c0cb8978d987597e0858c241d1a63fce5eb147348fd2be5a9e0c3092cbf237`.
Worker-closure digest:
`7957046e870eca1a8ccd53cd9d611f3a6838abc2accbff78f990d38a4f1fbd22`.

Capture exit 0 requires confirmed exact-instance cleanup. Separate read-only
recovery of that same Sandbox returned exit code 0 and identical candidate
runtime/directory facts, independently establishing it was stopped. A mistyped
image digest on the first readback command was rejected locally before the
Sandbox lookup; the corrected command used the capture's exact selection.
The complete closed capture and readback records are retained in
`evidence/modal-inference-qualified-d6e29ed.json` and
`evidence/modal-inference-qualified-d6e29ed-readback.json`.

This is a real CPU package qualification, not a GPU/model/chat result. No
Volumes, Secrets, GPU, model download, or external endpoint were attached.
The five-minute Sandbox timeout and network block were unchanged. The outer
candidate label remains intentional: fresh training, authenticated native read
qualification, consumer composition, GPU startup, one chat turn and that
session's cleanup still remain. The earlier no-ID failure is retained separately
and is not retrospectively rewritten as a confirmed cleanup.
