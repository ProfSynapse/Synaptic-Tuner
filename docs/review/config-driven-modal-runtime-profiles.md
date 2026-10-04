# Configuration-driven Modal runtime selection

Status: implemented and independently reviewed, 2026-10-04. No new model or GPU qualification is claimed.

## Contract

The recipe's `job.runtime_profile` selects a named runtime profile. The profile retains its exact model/revision/method allowlist, digest-pinned image and measured distribution inventory. An optional `runtime.packaged_build_profile` names a sibling build-profile directory below `Trainers/image_profiles`; it is required for the Modal packaged-training path, not for local-only profiles. It is a restricted name, not an arbitrary filesystem path or executable command.

Planning and execution must resolve the same configuration-driven selection. The selected runtime profile and build material must agree on the immutable base image and intended model/revision/method compatibility. Execution revalidates the profile and build intent retained by planning before cloud effects. Missing, mismatched, changed or escaping selections reject; no model-specific fallback is allowed.

The existing 4B profile declares its existing packaged build profile in YAML. A future model uses another explicit profile and build configuration; it does not require a Python switch, name-to-path map or new launcher. Declaring candidate compatibility permits a scoped qualification attempt only. The candidate is not a successfully qualified model until the exact runtime/model/hardware smoke provides that evidence.

## Verification scope

Provider-free profile/recipe regressions passed 58 tests on Windows, covering arbitrary profile selection and invalid or mismatched configuration. Five targeted Linux standalone checks passed: three stale plan-binding rejection cases, the public CLI prepare/submit/verify/download fake-provider flow, and CPU-only qualification without GPU start. The larger Linux suite was interrupted after passing progress because its fixture changed during that run; it is not claimed as a completed regression pass. Independent review found no remaining selection defect. Skill synchronization and scoped whitespace checks passed. Existing credential, source, inventory, release and one-shot submission checks remain intact.

The working tree already contains unfinished continuation support; those edits were preserved locally and must not be accidentally included in a paid qualification release. The changed host profile/recipe/runner files are not members of the existing runtime source-lock manifests, so this slice did not refresh their hashes. Exact release review, fresh runtime preparation and CPU/GPU qualification remain necessary before claiming a new model is qualified.

## Artifact retention and remaining qualification

The approved A100 80GB choice requires an artifact policy large enough for the intended rank-32 FP32 9B adapter. The generic Modal training policy now admits at most 512 MiB per artifact and 768 MiB for the complete five-artifact set. The producer, packaged worker, native reader and standalone downloader use that shared policy; the example consumer applies the same total bound. The read-only packaged-call diagnostic reports against matching limits. These are finite transport and retention bounds, not a model-specific exception or a change to the adapter representation.

Twenty focused provider-free tests passed on Windows. They cover a declared adapter above the former 192 MiB limit without a large allocation, rejection at one byte above the new per-artifact bound, rejection above the aggregate bound, diagnostic metadata, example verification and host composition. The standalone download test asserts the shared per-artifact bound. A broader local suite reported 519 passed, 558 skipped and 30 diagnostic failures: 29 required the unavailable `modal_proto` package and one mocked CLI returned `LOCAL_UNAVAILABLE`; none establishes a live provider result. Independent review found no artifact-policy defect and independently passed the 20 focused tests plus 10 metadata tests. The candidate 9B profile still needs exact source/release review and live model/hardware qualification before promotion; no new GPU qualification is claimed.

For evaluation diagnosis, retain bounded optional generation finish reason and token usage when the provider supplies them. Those fields can help explain why a response stopped; they do not measure writing quality or replace review of completed text.
