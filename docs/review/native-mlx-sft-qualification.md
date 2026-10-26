# Native MLX SFT qualification — 2026-10-26

The recovered API v1 work now has two successful bounded SFT mechanics
checks on the owner's Apple M4 MacBook Air: raw full-sequence and assistant-only
prompt/completion. This is exact-run hardware evidence;
ordinary native capability and Syntunia training remain gated.

## Execution and custody

Execution source was clean pushed commit
`7a8f36ec1b52f1201b25a35d962a576d74ac5d09`, branch
`feat/native-mlx-consumer-r3`. It includes the independently reviewed parser fix
for the pinned MLX writer's `__metadata__: null` safetensors headers. This report
and the accompanying test isolation repair are later changes; they do not change
the source identity of the observed run.

The existing isolated environment remained CPython 3.11.14, MLX/MLX-LM 0.32.2,
with the original 50-package lock plus reviewed four-package host addition.
Combined lock SHA256:
`7c167018ee6bd61a3609971e3f834082c798e7bd2a3fa0c5c2564887da4fde4f`.
Live recapture/check reproduced runtime capture
`81ad63f7c3933affc94a254192d79d79bb6d1c7fbdc248c44a6467d6a03d4c26`.
Hardware remained Apple M4 / Mac16,12 / 24 GiB / macOS 27.0.1 build 26A434.
Model/tokenizer remained `mlx-community/Qwen3-0.6B-4bit` at revision
`73e3e38d981303bc594367cd910ea6eb48349da8`.

Each admitted start uses an external root signature over its exact public
compiled workload, source/runtime/hardware/model/data selection and fresh
run/request/attempt/grant/state identities. The signing key stays on the Linux
host; existing named authority secrets remain in the Mac host and outside the
isolated trainer. The worker limit is three updates, 300 seconds plus five
seconds termination grace, 1 MiB per output stream, 64 MiB per artifact and
128 MiB aggregate. Historical consumed/uncertain attempts remain preserved.

R60 rejected before state creation because its signed expiry was more than one
hour ahead. A read-only check authenticated the same selection/signature and
confirmed the invalid window. Its records remain unchanged. R61 used fresh
identities and a valid shorter window; it did not replay an uncertain call.

## Raw full-sequence run r61

Run `f506602c-c341-44cc-bfc5-ac376daac3f7`, attempt
`a964b4d3-4109-42c9-9c84-3f0c272f5e29`, used the original synthetic publication
`0e4951bb2e54ed0985e6ca0d4c9940f0f5d75ccb78821c14544ef44897580cbe`:
three training rows and two validation rows, full-sequence labels, no template
arguments, batch one, sequence cap 128, seed 17, two final q/v LoRA layers,
rank four / scale eight and constant AdamW learning rate 0.0001.

The public start was accepted, public observation reported `succeeded`, and
public verification returned `verified: true`. The public downloader delivered
all five required roles. A separate Linux readback checked every delivered
size/SHA256 against the public verified manifest, parsed the adapter archive,
and recomputed the initial-to-final tensor delta with the native verifier.

- Actual optimizer updates: **3**.
- Changed scalar parameters: **40,960 of 40,960**; delta L2
  **0.03652545480504223**.
- Saved-final-adapter SHA256:
  `3f2caf022323be28ccd030226fd10debab98067a6937f2dc05108024dbd564fa`.
- Separately reloaded saved-adapter held-out loss: **4.3157453536987305**,
  two validation rows / 37 supervised tokens.
- Public payload total: **16,235,473 bytes**; native model role is an MLX
  adapter archive, not merged weights or a PEFT conversion.

Public manifest hashes:

```text
final_model       337920  0bc7a522168c9d85586181439b1d504104aaf74109412666fc91414a049be7c4
tokenizer       15892480  1a0d8f59e96a5aeae51232d71dd2ccb44ed32d8f2afbcec3af251a84b96bce1e
training_lineage   1078  96735b00de8a89cda1579ef9c8ad1e970a9b15b7433c1992bd3261729a2bc048
training_metrics   1214  e3529678c03ab713fcc4fb7e2416a9289e22d211cf7a25b41e6ed69fee36f764
workload_record    2781  ba67ec4c03c7224692066a57ca6672001eb0f8e98f294446068145c82f386bf7
```

## Assistant-only prompt/completion run r62

Run `c68eea51-46e1-4f7f-96a2-64f68f9490d4`, attempt
`3d1b4c3d-e43b-49bb-8c12-6e9d33712588`, used the original synthetic messages
publication `d22b95afbf5e415fa6a363381fda7714cfd0d7817d3b98a5e44e3149250ee9a0`,
three training rows and two validation rows. Its distinct admission retained
the same execution source/runtime/model/profile and finite training bounds.
The public compiler and verified lineage retain `assistant_only_loss: true`,
`prompt_render: prompt_completion`, and `chat_template_kwargs:
{enable_thinking: false}`.

All five complete source rows and the retained tokenizer doctor were reviewed
before start. The doctor reports zero hard failures, truncation or dropped rows,
and 43 supervised tokens. Its previews mask the entire prompt, including the
pinned template's empty thinking scaffold, and supervise the final assistant
answer through end-of-turn token 151645. Current publication/tokenizer hashes
and unchanged shared/native preprocessing source preserve that earlier review.
No generated writing-quality evaluation is claimed.

Public observation reported `succeeded`, verification returned `verified: true`,
and all five public roles were downloaded. Independent Linux readback matched
every manifest size/hash and recomputed the saved adapter delta:

- Actual optimizer updates: **3**.
- Changed scalar parameters: **40,960 of 40,960**; delta L2
  **0.037091878184678954**.
- Saved-final-adapter SHA256:
  `4eee052cb43eda79c1a8477c0485c51e1cf946076a97dd87b77266c45876752a`.
- Reloaded saved-adapter held-out loss: **3.949827194213867**,
  two validation rows / 17 supervised tokens.
- Public payload total: **16,235,561 bytes**.

The raw and messages losses supervise different tokens and are not a
before/after model-quality comparison.

```text
final_model       337920  2cfdbd70d189b562611d4c83d3744249924c20ae67fa671036b82431095ae82c
tokenizer       15892480  668e35f7bc5da35ff0a36a277eb8c0f37f4c004a75bee3b4ece92d8f4690bb36
training_lineage   1098  42d71184d950e2a000a931a3aca9e49ef8a4ce45c6046049a3f6a9e28207e013
training_metrics   1214  c9cd9ff87d0d17461f1cb5c28ed563f975df01b41a5416d50a660096519553da
workload_record    2849  e9e442e55434446f66903cddcad544d6d370c8c012480bfe28cbec36065488aa
```

## Test isolation and remaining gates

The prior PR269 CPU core CI ended with `OSError: [Errno 27] File too large`
while pytest wrote its JUnit report. A targeted reproduction showed that two
cleanup tests calling the real guardian inline changed pytest's hard
`RLIMIT_FSIZE` from unlimited to 1 MiB. The test now records/asserts those limit
calls without applying them to pytest. The real subprocess tests still exercise
the production guardian. All 108 focused process/host/execution/publication
tests pass, JUnit output completes, and the parent file-size limit stays
unchanged. This is a test isolation fix, not a production runtime change or
a claim that every historical CI failure has the same cause.

These two synthetic mechanics results do not establish model quality, broad model
or memory compatibility, cancellation/restart recovery, native GRPO, general
reliability or Syntunia product integration. Public logs/cancel/reconcile stay
disabled. Independent qualification admission, lifecycle acceptance, final
release branch integration and Syntunia's actual training host joins remain
separate work. Do not fabricate a `NativeQualificationV1` from this document
or historical hashes, and do not retrain to repair documentation or readback.

Private original admissions, command results, five delivered role files and
independent report are retained under
`~/.cache/synaptic-tuner-release-proof/resume-20261026/native-raw-r61` and
`native-messages-r62` on the
Linux host, with the original attempt state and public delivery on the Mac.
Nexus workspace `Fine-Tuning Lab`, task `T-0c6ae8a8`, records the continuation.
