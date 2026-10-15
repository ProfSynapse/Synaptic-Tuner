# Public Docker SFT qualification host

This dedicated host branch is not an engine PR. Its real engine Gitlink is
038a62984a4745574d1eaf739a77272a62bb6deb, with the declared
`release/stable-qualification-fixes` branch hint (PR258). Local Git
inspection proves the actual checkout/gitlink and clean source; it does not
prove this host has been pushed. Root reviews/pushes the host and verifies exact
remote refs before source authentication or provider effects.

The unchanged eight authored Question/Answer documents remain at the historical
`data/hf-sft-v1/sources` path. The name records their origin; these files are the
shared raw-text fixture, not an active HF job configuration. Public ingestion and
v1 preparation produced the retained publication under `data/hf-sft-v1/prepared`.
Six rows are training and two validation. All original question/answer text is
present. This v1 full-token-loss lane is not messages/v2, completion-only masking
or a model-quality evaluation.

Retained bundle digest: cbad3e6be421470a8f3480944f95e52f151ec315d0914472fee27e2449265a1b.
Prepared semantic identity: 7aa26aba079ac1210466c5c2cdf747f00c764b5bc4ed366e2f67af7ffa0d3770.
Original JSONL: 2,821 bytes, SHA256739b3df17bf7fc39d5c9487136a633e8ad2703e700271e224756df8074d2c361.
The retained source SFT config selects Qwen/Qwen3.5-4B and tokenizer revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. The source fixture retains its three-step duration and explicitly uses the
existing adaptive precision policy. Unsupported BF16/FP16 configuration fields
are removed in this new host commit; earlier commits preserve the original
configuration. This fixture is not a new completion/cancel request.

New Docker inputs are private and create-only. The checkout-owned
`examples/training_qualification/prepare_docker.py` alongside the accepted
installed engine compiles actual source-authenticated material and profile, then
uses a separate explicit pinned-model preparation command. The historical
v2-only image profile is not rewritten or reused as v1 authority. Actual derived
image/profile/report/release and final wheel commitments must be reviewed before
publishing ready consumer configuration. Normal execution uses public
TrainingAPI/RunsAPI, never a legacy local-run shortcut.

The selected desktop is DESKTOP-2A4U5KR Ubuntu WSL with an explicit local Unix
Docker endpoint and real mount paths. Local declarations and a CPU filesystem
probe do not establish GPU/model/update, cancellation or physical termination
qualification. Completion and cancel candidates have separate unused attempts;
unknown stage/submit/cancel outcomes never authorize replay or state deletion.
Original named host/source/read keys remain private and outside tracked files.

`configuration/hf-sft-v1` keeps original ingestion/preparation/config and unbound
HF preparation templates for provenance. Their bucket/quote/billing/host-secret
placeholders are inactive for Docker. No HF account or billing selection is
implied by this host branch. Source/fixture setup itself performs no image build,
model download, GPU training, submission or cancellation. The separately
approved public base pull is external retained evidence, not a claim by this repo.

Git retains the historical prepared publication bytes, not private filesystem modes. The fixture test verifies an exact private copy; execution must create a fresh private publication through the supported preparation workflow.

This fresh local host selects the diagnostic candidate for review. Publishing this host does not itself grant execution authority. Earlier consumed runs and private state remain preserved.

The current candidate changes the SFT core and its authenticated closure. Image B
and release b186cc00 remain historical evidence; they do not attest this repaired
runtime. A fresh derived image with the accepted current wheel, actual capture,
verification and CPU result is required before new training. Both independently
verified model caches and all earlier claims remain preserved. Fresh run IDs,
source/material bindings and supported explicit cache mapping are required;
never replay an earlier consumed or uncertain attempt.
