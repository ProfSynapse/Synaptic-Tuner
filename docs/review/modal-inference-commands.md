# Exact Modal chat command retention

Status: locally qualified, 2026-09-10. Engine-only provider-free work;
no Sandbox, serving grant, inference runtime lock or live qualification.

## Boundary

`ModalInferenceCommandBinding` retains exact immutable Foundation command bytes
and the complete canonical chat preparation snapshot. It accepts STAGE/SUBMIT
only and reconstructs their preparation, executor, payload and predecessor
correspondence. It is content validation, not authentication. In particular,
neither the snapshot's configuration/quote tags nor its image/lock commitments
prove current source admission or an independently inspected inference runtime.

`retain_modal_chat_command` uses a consumer-owned catalog and complete-content
authentication authority. It authenticates a reconstructed candidate before
publication, reads first, publishes only if absent, and authenticates and compares
the retained result. A conflict, missing reread or ordinary collaborator failure
fails closed without overwrite, internal retry or Foundation/provider dispatch.
`load_modal_chat_command` authenticates exact retained content by command digest
without writing. Both helpers guard input and owned snapshots across callbacks;
ordinary failures expose only `modal_inference_retention_invalid` while process
control exceptions remain process control exceptions.

Recovery validates retained content without rerunning expiring preflight evidence
or inventing new source admission. The consumer must compose the original fresh
source/workload/configuration/quote admission with its existing durable grants
and Foundation evidence. A matching predecessor projection is not authentication
of the predecessor's stage record or receipt. Catalog publication is not a grant
and does not consume or recreate provider submission authority.

CANCEL is deliberately excluded: retaining a provider target string alone would
not establish ownership of a Sandbox lease. No new catalog implementation,
database, authority framework, CLI or public export is introduced.

## Sequencing

The runtime-lock review found no complete inference worker/bootstrap inventory
and no inspected inference image digest, Python executable identity or inference
dependency closure. Consequently this slice advances exact command retention
before freezing that lock. It does not relabel the training lock or treat the
checked-in vLLM image tag as a verified inference pin. The remote worker,
authenticated Volume-to-mount admission, executor/transport, owned lease, separate
runtime lock and bounded session adapter remain required before live serving.

## Qualification

Source implementation is frozen and independently reviewed without a blocker.
The original 37 acceptance cases passed in 100.15 seconds. Twenty additional
lead-authored cases passed in 38.74 seconds, including independently rehashed
invalid snapshots, same-command/different-content conflicts and callback
mutation. The final independent run passed all 57 command/retention cases plus
35 existing preparation cases: **92 passed in 240.91 seconds**.

The combined regression selection passed **2,495 tests in 565.00 seconds** under
isolated CPython 3.12.9 / pytest 8.4.2, with no system site packages, Modal or
Torch and automatic pytest plug-ins disabled. This is the previous 2,438-test
selection plus 57 cases, not the entire repository suite. It covers the existing
coordinator/Foundation, provider-neutral/public API, training/runtime, Docker
provider, inference, selected Evaluator/client callers and Modal provider tests.
The additional packaging regression described below was added after that run
was collected; its five-test contract module separately passed in 0.33 seconds
and was independently repeated. Do not report 2,496 as a single full-suite run.

An initial real-fixture probe reproduced a wrapper-size defect: the preparation
snapshot was 12,178 bytes and the valid SUBMIT command 4,225 bytes. Each fits the
unchanged 16 KiB Foundation document limit, but their combined binding does not.
The fixed-shape envelope now has an exact bound of two 16 KiB components plus
99 bytes of schema/key overhead (32,867 bytes total). Both components still use
the unchanged Foundation canonical parser and limits. These measurements
are synthetic fixture content, not credentials or live provider evidence.

A dependency-only install of the exact `bd6aff309f913d495d38079314e4cb2a319c500c`
wheel exposed a pre-existing packaging omission: importing
`Evaluator.verified_vllm_chat` reached `vllm_client` then `openai_compat_client`,
which imports undeclared `requests`. The broader test environment already
installed that dependency and masked the omission. Package metadata now declares
the existing CI-compatible `requests>=2.32,<3` requirement; a regression assertion
and a dependency-only installed-wheel CI step prevent the same masking.

The corrected source commit is
`9764d85755d45905dcc55e184bb1421ba0c79085`. Its exact Git archive produced
`synaptic_tuner-1.1.0-py3-none-any.whl`: **2,058,529 bytes**, SHA-256
`4aabb69af67bb9cea16ad1d45230d8736082502aadb9cf614dfd872e0d029775`.
Independent audit verified 721 unique ZIP/RECORD entries, every recorded size
and hash, and all 713 Python members byte-for-byte against the archive. All 713
Python payloads are also unchanged from the prior runtime-tested source wheel;
the correction changes dependency metadata, CI, its regression test and this
note, not runtime code. METADATA declares `Requires-Dist: requests<3,>=2.32`.

Installed into a fresh environment containing only the wheel and its declared
dependencies, the replacement passed all **40 engine/Evaluator imports**, both
packaged resource checks and four credential-free `inspect.Signature.bind`
checks. The probe used isolated Python from a neutral directory and confirmed
the imported module belonged to that environment. Modal, Torch, pytest, NumPy
and pandas were absent. The earlier wheel's missing-requests failure remains
part of the evidence; it is not reclassified as a successful installed test.

The 97-member training runtime lock and separate 66-member offline worker closure
(679,487 payload bytes) both report `CURRENT`; no training inventory or runtime
pin changed. On this mixed Windows/WSL worktree, the offline closure check needed
explicit local `GIT_DIR`/`GIT_WORK_TREE` to address its Git oracle's UNC-path
lookup; no `PYTHONPATH` was used. Canonical skill mirrors match. The independent
wheel scan was filename-only and found no credential/private artifact indicators;
it makes no content-level secrecy claim. No provider SDK, cloud resource, model
download, training job, GPU, push, merge or EHR change was part of this slice.
