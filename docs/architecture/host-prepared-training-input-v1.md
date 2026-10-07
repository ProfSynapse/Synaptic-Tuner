# Host-Prepared Training Input v1

`TrainingAPI.prepare(source, config)` is an additive public-v1 verb. It precedes
the existing `load -> resolve -> plan -> preflight -> start` flow and does not
change the meaning or signatures of those verbs.

The input is a host-only local path or immutable one-use upload snapshot plus a
declarative `TrainingPreparationConfigV1`. A normalizer receives only its own
closed typed config. No normalizer config is copied into the public result. The
result separates two concerns:

- `PreparedTrainingInputV1` is canonical and serializable. It contains only the
  `prepared://sha256/<semantic-digest>` identity and the canonical training
  request using that reference. Construction reparses that request through the
  strict `TrainingInputV1` parser and requires byte-for-byte canonical JSON.
- `PreparedTrainingInputResultV1.retained_source` is process-local staging
  authority. The result and source objects reject serialization. Paths,
  filenames, stream handles, upload handles, and source prose never enter the
  canonical prepared value.

Before returning, preparation verifies the format-specific publication,
recomputes its byte count and SHA-256, opens the retained source, consumes that
lease fully to EOF, and requires those bytes to equal the verified snapshot.
An unavailable format or fabricated normalizer result therefore fails during
`prepare`, not later during provider staging.

`PreparedTrainingInputIdentity.execution_binding_fields()` is the sole Slice A
conversion boundary. It maps `ref`, `revision`, `content_digest`, `size_bytes`,
and `format` to the packaged execution binding's corresponding
`prepared_input_*` fields without importing a runtime or provider module.

The built-in `synaptic.dataset-prep/v1` normalizer accepts an admitted local
normalized-bundle path and a closed, path-free existing dataset-prep v1 or v2
config. It rejects linked/reparse bundle members and hardlinks. Allowed source
root objects are retained by filesystem identity and rechecked before and after
normalization; replacing an allowed root, even by the same account, fails
closed. This is identity checking, not protection from an already-authorized
same-account process mutating ordinary source file contents between OS calls.
It delegates normalization, verification, content-addressed publication, and
collision handling to `tuner.dataset_prep`; it does not create a second
publisher. The currently registered `syntunia-sft-row/v1` and
`syntunia-sft-row/v2` identifiers are built-in dataset-prep format adapters,
not assumptions in the generic preparation architecture. Other typed
normalizers and format verifiers can be registered without changing the public
contract.

The prepared root requires an explicit host privacy authority. The included
POSIX authority creates or validates a user-owned mode-0700 root and rechecks
the root object's filesystem identity throughout preparation. Windows does not
treat `chmod` as evidence of a user-only ACL, so the default service refuses to
prepare there unless the host injects an attestor that proves the root's ACL
and identity. The remaining Windows ergonomics follow-up is a platform adapter
backed by native security-descriptor APIs; this slice intentionally does not
shell out to mutate ACLs. The trust boundary is the local OS account and the
injected attestor: neither mechanism defends against a process already holding
equivalent account authority.

Preparation is provider-neutral. It contains no Modal, Hugging Face, RunPod, or
Docker identifiers or storage behavior. Provider adapters later stage the
retained source using its atomic one-use verified stream lease. Host-only
sources, leases, upload snapshots, and results reject copy, deepcopy, and
pickle serialization.
