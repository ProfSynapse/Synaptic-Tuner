# Native MLX public lifecycle: bounded M4 evidence

Observed 2026-10-27 on execution source
`e29c7da8cdb0f5f6ea29ecfe6b1edba397d525b1`. Eight separately signed,
single-use attempts exercised public progress, normal completion, cancellation
and abrupt caller loss followed by cold reconciliation for SFT and GRPO.
This is engine integration evidence for the exact fixture below. Final R6
independent evidence/package/claims review accepted `9524bae2`; all 18 checks
passed before PR #279 merged at `e2e8cd4f` on 2026-10-27. The later retained-read
gate also passed, as recorded below; execution remains bound to `e29c7da8`.

## Selection and custody

- Apple M4, 24 GiB, Darwin arm64, macOS 27.0.1 / build 26A434.
- Existing separate private CPython 3.11.14 / MLX 0.32.2 / MLX-LM 0.32.2
  runtime. All 54 selected wheel archives, install receipt and private custody
  passed revalidation. Original runtime lock:
  `7c167018ee6bd61a3609971e3f834082c798e7bd2a3fa0c5c2564887da4fde4f`.
- Model/tokenizer: `mlx-community/Qwen3-0.6B-4bit`, revision
  `73e3e38d981303bc594367cd910ea6eb48349da8`; complete existing snapshots
  revalidated, without download, conversion or installation.
- Fresh capture:
  `39c6dd15acd7c3460a3f2a989de75f1c137e9021443f813e5e2e06d4af21684e`.
  All 757 selected source members matched exact Git bytes. Runtime trees,
  runtime identity, native-library graph and OS identity matched the preceding
  independently reviewed GRPO capture. The exact OS shared cache remains an
  explicit trust boundary; capture alone is not execution authority.
- R4 independently accepted the worker/public source and R5 accepted executable
  host-workflow parent `cdf24699`. The execution commit changes only the Darwin environment test
  and skill guidance; executable engine/host bytes are identical to R5.

The first Darwin CPU run passed 184 controls with one failure and one skip.
Bounded direct-interpreter and shell probes identified initialized Darwin CF
encoding and fixture-shell `SHLVL` names absent from the test's allowlist.
The test-only correction checks their numeric/user identity only on Darwin and
rejects parent-variable/credential sentinels. Production environment construction
was unchanged. The corrected final-source run passed **186 controls, no skips**,
including high-FD startup after raising only the test child's soft descriptor
limit. Pytest's hard limits were preserved. Five real MLX SFT/GRPO/reference-loss
numerical tests also passed without skips. No historical failure was reclassified.

## Admitted operations and outcomes

Every attempt had a distinct run/request/attempt, unused private state root and
root signature over the full source/runtime/hardware/model/input/workload and
scenario selection. The signing private key remained server-only; existing host
authority secrets remained Mac-only. No admission was renewed or replayed.
Each worker was limited to three updates, 300 seconds plus five-second cleanup
grace, 1 MiB per output stream, 64 MiB/file and 128 MiB aggregate artifacts.
Progress allowed 32 numeric records and a 128 KiB journal. Host observation had
330 seconds; cold recovery had 30 seconds.

| Method/scenario | Run | Public terminal | Guardian reason | Live records |
|---|---|---|---|---:|
| SFT complete 1 | `0afe1c61-0660-48fe-bd09-838c9f85fe07` | succeeded | worker_exited, 0 | 8 |
| SFT complete 2 | `4f104531-b55a-44ce-9431-3bae56dea41c` | succeeded | worker_exited, 0 | 8 |
| SFT cancel | `982961a1-6229-49dc-bcb9-bc7b0ec3e3d1` | cancelled | cancel_requested, -15 | 3 |
| SFT caller loss | `fbd0afc1-c6fa-43c4-9af6-ba22b8ea2905` | failed | owner_lost, -15 | 3 |
| GRPO complete 1 | `6664e5e3-9078-4421-ae0a-cadf8c9e8f7a` | succeeded | worker_exited, 0 | 6 |
| GRPO complete 2 | `7adf35ef-aa85-4cc4-8861-5048eb5a9cf4` | succeeded | worker_exited, 0 | 6 |
| GRPO cancel | `3bb04f20-e5fd-48f5-a04b-64b959ed7219` | cancelled | cancel_requested, -15 | 2 |
| GRPO caller loss | `8cb03b17-4596-4aa6-a120-6bc0f458ffe8` | failed | owner_lost, -15 | 2 |

All eight guardian outcomes confirm termination. Each retained exactly one
process claim. Every live record was delivered between public RUNNING outcome
reads. Messages contain fixed phases and numeric metrics; timestamps are stable
host-first-observed times, not GPU event timestamps. Cold public logs preserve
the original entries and timestamps. This proves observed delivery, not universal
freshness or completeness of every possible future stream.

Cancellation was requested once after the first reported training update.
The public `cancel_requested` acknowledgement was retained separately from the
later confirmed `cancelled` outcome. No persisted PID was used to signal.
Caller-loss cases first retained the public progress trigger, then exited the
actual host process with code 86 without Python/session/store cleanup. A new
process called public reconciliation and read the original attempt's authenticated
`owner_lost` termination. Public `failed/native_worker_failed` is the intentional
closed result for an interrupted training attempt, not recovery failure.
No training was adopted or restarted. This is recovery of known attempt state,
not optimizer/checkpoint continuation.

## Training and artifact checks

SFT reused the genuine HF conversation publication
`38f84d342479589e7de6220c5cc7359418543e26b67e4e3c9816bfba23a27483`:
100 training / 20 validation rows, exact sequence 512, batch two, seed 17,
assistant-target masks and `enable_thinking=false`. Three updates consume six
seeded training rows; **this lifecycle run is not a full dataset pass**.
Both completions separately reloaded the saved final adapter and evaluated all
20 validation rows / 650 supervised tokens, finite loss `2.1788835525512695`.
All 40,960 adapter parameters changed; independently recomputed delta L2 is
`0.03639340059580386`. The two saved adapters are byte-identical for this fixture.

GRPO reused the genuine two-prompt mechanics publication
`45c53c9395bf579b59af0172ce0368c4ac2bac2a8457d97d5fd0d4a15bdaf074`,
batch/group 16, seed 17, prompt 64 / completion 8 / sequence 128, configured
blue-word reward and three updates. Both completions retained 48 reward calls,
102 generated tokens and three groups whose reward variation survives float32
conversion. All 40,960 parameters changed; independently recomputed delta L2 is
approximately `0.0407861391343181`. Saved-final-adapter rollout-replay loss is
finite at `-0.013764187693595886`; it is not a held-out quality measure.
The two saved adapters are byte-identical for this fixture.

All four normal completions passed public initial verification and delivered
all five bounded artifact roles. All **20 role hashes and sizes** matched the
verified public manifests on independent local readback. Adapter changes were
recomputed from initial files and delivered final tensors using separate scalar
decoding/arithmetic. Saved validation identity matches each delivered adapter.
Two repeats demonstrate this bounded workflow's observed repeatability, not
success on arbitrary future invocations.

## Host integration and limits

Use the [host contract](../architecture/native-mlx-host.md),
[lifecycle design](../plans/native-mlx-lifecycle.md) and existing
[config-first host workflow](../../examples/training_qualification/native-host.md#config-first-lifecycle-qualification-candidate).
The versioned opt-in requires exact policy/codec, a host-owned session and
independently authenticated lifecycle qualification or restricted admission.
Metadata alone enables nothing. Ordinary qualification remains a host-owned
authentication decision; no accepting default authority or global native
registration is shipped. Legacy v1 stays synchronous with lifecycle flags off.

Darwin CPU controls cover lost submit/cancel acknowledgements, completion races,
prefix tampering and unknown effects. The actual ML attempts above qualify
normal delivery, public cancellation and abrupt caller-loss recovery; they do
**not** add a separate actual-ML acknowledgement-loss experiment. Preserve that
distinction. Missing/corrupt terminal evidence remains uncertain, with no
adoption, guessed absence, replay or stored-PID signal. Host/guardian/machine
failure and hostile descendants escaping the owned group remain outside this
proof. Other models, rewards, OS builds and runtimes require their own admission.

These observations do not establish model quality, current Nexus tool compatibility,
full desktop distribution, UI readiness, installer signing or a Syntunia deployment.
Earlier SFT/GRPO/evaluation evidence retains its original source bindings.
The eight start admissions expired at `2026-10-27T22:39:31Z`; the initial cold
reads above preceded expiry. At actual Mac time `2026-10-27T22:39:33Z`, root
completed fresh original-authority post-expiry reads: all four normal runs passed
public `reverify`, and all eight public outcomes/manifests/progress entries and
timestamps exactly matched the earlier cold reads. Fresh SQLite snapshots still
showed one process claim per attempt. No retraining, renewed admission, adoption
or stored-PID signal occurred. These reads authenticate retained results; they
do not extend authority to start a new attempt.

## Retained evidence

Private evidence root:
`/home/profsynapse/native-mlx-lifecycle-20261027-r1/mac-e29c7da8-r2`.
It contains canonical selections and admissions/signatures, capture/source checks,
CPU/numeric results, per-attempt public reports, cold logs/outcomes, read-only
SQLite snapshots, delivered roles and initial adapter tensors. See
`selection-independent-check-r1.json` and
`qualification-independent-check-r1.json`. No private signing key or host secret
was copied into that evidence bundle.

Final wheel/sdist and installed-byte equivalence passed independent R6 review,
including all 27 native modules and the base-only consumer without ML/provider
SDKs. The reviewed docs head `9524bae2` and merge `e2e8cd4f` have identical trees;
all 18 checks passed. Post-expiry evidence includes
`post-expiry-independent-check-r3.json`, `post-expiry-claim-counts-r3.json` and
each attempt's retained reports/SQLite snapshot. This completed the bounded
engine integration. Subsequent [release metadata](../releases/1.1.0rc1.md)
does not relabel the execution source or establish a full desktop release.
