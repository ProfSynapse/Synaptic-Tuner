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
