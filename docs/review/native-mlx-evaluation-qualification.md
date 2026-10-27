# Native MLX: bounded M4 generation and held-out comparison

Observed 2026-10-27 on execution source
`edc4c7aa10e1e26965e36e3039fc2e25d1bfe330`, independently accepted by source
review R3 before execution. The pinned base and two existing SFT adapters each
completed the same 40 cases. This qualifies the recorded bounded generation
workflow; useful tool-call behavior remains unqualified.

All three models passed **0/20 historical command checks** and emitted no parsed
tool calls on those prompts. Each adapter lowered reference loss on its own
dataset. Lower loss did not establish successful tool use. No new training was
submitted to obtain inference authority.

## Exact execution and inputs

- Apple M4 / Mac16,12 / 24 GiB, Darwin arm64, macOS 27.0.1 build 26A434.
- Existing separate private CPython 3.11.14, MLX/MLX-LM 0.32.2 and unchanged
  54-package lock
  `7c167018ee6bd61a3609971e3f834082c798e7bd2a3fa0c5c2564887da4fde4f`.
  No package installation or original-runtime modification was needed.
- `mlx-community/Qwen3-0.6B-4bit`, model/tokenizer revision
  `73e3e38d981303bc594367cd910ea6eb48349da8`. Model snapshot digest
  `945b17f849887f1f52e62b156e2f1ad821e2e6218f57f97e5d52ca73a83b46fe`;
  tokenizer snapshot digest
  `57c46cb5e041f8f45254273fcd1dd8c4fe7ca767f89bdc4c8d125b7187614b51`.
- Fresh runtime capture
  `2ded31c8f45ad8e4d8b3c7f741dd416ec4780c1c64fa3bb38a81bb29536f1ee6`.
  All 751 captured source members matched the execution source. Runtime trees
  and the runtime record matched the preceding accepted GRPO capture. The exact
  OS shared cache remains an explicit trust boundary; a capture is not authority.
- Five real Apple numerical tests passed: three SFT loss/gradient cases, GRPO
  objective/detachment and evaluation reference loss. Both existing tokenizer
  profile commands passed all 120 original rows, with zero failures, on final
  execution source before capture.

Cases use all 20 declared validation rows from each existing immutable
conversation publication. Every source had 100 training and 20 validation rows;
no resplit or target repair was performed. The full prefix is retained and the
final assistant reference is separate from generation. Preparation and original
training details remain in the [dataset workflow](../../examples/training_qualification/native_sft_datasets/README.md).

| Material | SHA-256 |
|---|---|
| Conversation publication identity | `38f84d342479589e7de6220c5cc7359418543e26b67e4e3c9816bfba23a27483` |
| Nexus publication identity | `c6bca71a6463a8d89cf4bcac970985315b1f480e4d5f3d27050a81fb728b64c2` |
| Ordered generation cases | `3872dcf244ece06799f04948b6766ab94d8b72d4c7f88ce3daafdaa12f988ac1` |
| Scoring cases/assertions | `d22d27f854619c8716a5516c27175accc9181125e9661a6e55189380b85a3f4a` |
| Conversation model-role archive | `387adb5286f4b832f2c1957b0a3468a4dd2b2f43b56dc68383ef85ca29e5ec26` |
| Nexus model-role archive | `65cf63bd44d96695dd45a048707da47232882a1d659074a2130caf41c1bada63` |

The saved adapters originate from actual 50-update SFT runs on source
`d57dc55e3797c32761435980a41b0e05182e5aa1`: conversation run
`c392174e-25aa-47cd-bf69-96e58a559a1a` and Nexus run
`82bdd254-fe51-4708-81df-caf59db26697`. Both original model roles were verified
before copying. Generation reloaded and compared the exact saved adapter values,
not an in-memory training model. Their training evidence retains its own source.

The existing contamination checker found no exact duplicate prompts or threshold
violations against the 200 selected training rows. Maximum normalized word
8-gram containment was 0.3068 at threshold 0.5. This is not proof of broad
distribution independence. All 120 actual generation prefixes later matched the
original held-out tokenizer IDs exactly; reference token counts also matched.

## Admission, bounds and delivery

Each model had its own new private state, attempt, full selection and externally
signed generation-only admission. The original signing key stayed on the server;
the Mac received only public admission/signature/key material and checked an
external key pin. Ordinary native training authority was not reused.

| Model | Selection digest |
|---|---|
| Base | `54dfe4ba6aac69ff16f582ac619b8156af9536e06bc7f47d770dea63be21046d` |
| Conversation SFT | `52baa14f255aef0b236eba45093c9f7bff2b5f70c48f42efda8417631c2d3e8b` |
| Nexus SFT | `f0a8cec259558073455805fb985b132d58b8df651f7416a8ca5c37fede4b1cca` |

The named `mlx-m4-qwen3-06b-evaluation-v1` profile selected greedy decoding,
seed 17, `enable_thinking: false`, stop ID 151645, at most 2,560 prompt plus
1,536 completion tokens in a 4,096-token context. Bounds were 900 worker seconds
plus five-second cleanup grace, 1 MiB streams, 16 MiB per file, 8 MiB case
evidence and 64 KiB response text. The longest reference was 1,329 supervised
tokens; the longest prompt was 1,410. No truncation or dropping was permitted.

Every owned process exited zero with termination confirmed, 40 complete cases
and no training submission. Host readback and complete private export passed.
The original adapter admissions expired at 18:17:51Z and 18:20:23Z; fresh-process
readback at 18:21:46Z used those original signatures and retained the same result
digests, without loading a model or renewing start authority. The base admission
has its distinct 18:36:07Z start expiry; do not infer its expiry from adapter reads.

| Model | Complete export manifest SHA-256 |
|---|---|
| Base | `fa901caa9cc2a97352a12deab410795e4ceaa99ebde3760049b33cc7459e5f04` |
| Conversation SFT | `5681fccedf97cdbf35503ee9eb95915b519397159461183032c63a0f3135f989` |
| Nexus SFT | `118b4166a51f4fecaaa86e4d7fc79eb05bab5105936f9179353b5ba353f8e381` |

CPU scoring verified each original signature/selection, exact complete inventory,
member hashes and prompt/case join before using the existing evaluator. Portable
evidence retains an operator-owned-process witness; it does not independently
authenticate model execution as a provider attestation. Complete private
generations remain retained. No generated command was executed against a vault.

## Results and interpretation

Reference loss is causal, masked cross-entropy weighted by supervised tokens:
650 conversation tokens and 4,114 Nexus tokens per model. References never
entered generation. Compare these values within this evaluation; earlier SFT
validation reports use their own batching/aggregation and are not relabeled.

| Model | Conversation reference loss | Nexus reference loss | Historical command checks | Natural stops |
|---|---:|---:|---:|---:|
| Base | 2.183849 | 1.609131 | 0/20 | 40/40 |
| Conversation SFT | 1.908499 | 1.584412 | 0/20 | 40/40 |
| Nexus SFT | 2.089040 | 1.250325 | 0/20 | 40/40 |

All 120 generations stopped naturally; none hit the token limit. Generated token
counts were 4,511 / 3,372 / 3,408. Summed generation-only latencies were 39.80 /
36.47 / 36.67 seconds, excluding startup, verification, reference loss and
transport. These are observations, not completion-time promises.

Conversation assertions only require nonempty text and passed 20/20 for each
model. They do not score semantic quality. Historical assertions require the
declared old tool name and command string; equivalent quoting could fail them.
Here all 60 historical responses contained no parsed tool call and no literal
`useTools`, so the observed gap is broader than an exact-string mismatch.
Responses included prose about opening/checking files rather than tool calls.
This observation does not establish its cause.

The existing comparator reports matching inputs/controls, zero request errors and
no assertion-pass improvements, with no thresholds applied. Its aggregate 50%
mixes nonempty conversation checks with failed command checks and must not be
presented as model accuracy or semantic quality. Its GGUF-oriented quantization
label detector does not label MLX artifacts; all three selections here bind the
same verified four-bit base. This was an adapter comparison, not a quantization
study.

## Remaining gates and next experiment

The engine path completed this bounded cohort. Public logs/cancel/reconcile,
caller-loss/restart recovery, arbitrary models, ordinary native admission and
broader stable-v1 support remain gated. Full desktop distribution is excluded.
Final current CI and independent retained-evidence/claim review precede merge.

Useful GRPO reward readiness is not established. Greedy output supplies neither
sampled within-group reward variation nor current Nexus API correctness. The
existing native reward engine is bounded response-only regex V0; the earlier
color fixture proves mechanics, not practical tool behavior.

The next proposed experiment is a config-only, bounded contrast with an explicit
current tool schema/serialization contract and reviewed acceptable alternatives.
Keep the original publications unchanged, preserve held-out lineage, and treat
scaffold effectiveness as a hypothesis to test. Establish sampled useful reward
variation before further practical GRPO qualification. Do not hardcode the Nexus
wrapper or silently change the generic evaluator/runtime to make scores pass.

Operator evidence is retained under
`native-mlx-quality-20261027-r1/mac-edc4c7aa-r1`: original selections/admissions,
capture, numerical/tokenizer evidence, three complete exports/reports,
`comparison-r1.json` and `comparison-summary-r1.json`. Later documentation or
package checks do not change the execution source or admit new model effects.
