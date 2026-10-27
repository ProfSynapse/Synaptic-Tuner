# Separate native GRPO runtime candidate

`native-mlx-grpo-on-policy-candidate-v1` is an unqualified named source profile,
not an installed environment or execution grant. Create a new owner-private
runtime through the existing `native_runtime_preparation.py` workflow only after
parent/root review; never install into the qualified SFT environment.

`requirements.runtime.lock` is an intentional byte-for-byte copy of the reviewed
54-package SFT runtime closure (MLX/MLX-LM/Metal 0.32.2, CPython 3.11.14 arm64).
No installed `mlxrl` or `mlx-tune` package is selected. The attributed objective
is captured engine source, with its full MIT notice inside the Python module so
both source and wheel carry it. Identical package pins do not qualify GRPO or
authorize environment reuse. Original SFT locks are unchanged.

The existing native capture CLI must capture the fresh runtime, full native
libraries/OS boundary, and final pushed `shared`, `tuner`, `synaptic_tuner`
sources. `candidate.json` declares the selection; it deliberately has no invented
runtime/source digest or future signature. After real capture the host builds
an exact `NativeMLXGRPOProfile`, using `native_host.py profile-candidate` with a
reviewed proposal whose `method` is `grpo`. Include the eight extra GRPO profile
fields under `grpo_profile`; include exact compiled `NativeGRPOControls` under
`grpo_controls`. Existing inspect/admission-candidate/start/read actions apply.
The attempt kind is `grpo_on_policy`; `input_review` binds `prompt_rows` and
`publication_sha256` to the existing private GRPO publication. Proposal training
bounds and LoRA/AdamW controls retain the existing native consumer contract.

Read `docs/architecture/native-mlx-grpo.md` before proposing hardware work.
Independent source/CPU review and fresh root admission precede ML imports.
No GPU fit, quality, public progress/cancel/reconcile or qualification is claimed.
