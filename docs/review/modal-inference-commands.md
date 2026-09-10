# Exact Modal chat command retention

Status: implemented, broader qualification in progress, 2026-09-10. Engine-only provider-free work;
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
mutation. Both runs used isolated CPython 3.12.9 / pytest 8.4.2 with no system
site packages, Modal or Torch. No installed-wheel or full-selection pass is
claimed yet; those checks are in progress.

An initial real-fixture probe reproduced a wrapper-size defect: the preparation
snapshot was 12,178 bytes and the valid SUBMIT command 4,225 bytes. Each fits the
unchanged 16 KiB Foundation document limit, but their combined binding does not.
The fixed-shape envelope now has an exact bound of two 16 KiB components plus
99 bytes of schema/key overhead (32,867 bytes total). Both components still use
the unchanged Foundation canonical parser and limits. These measurements
are synthetic fixture content, not credentials or live provider evidence.
