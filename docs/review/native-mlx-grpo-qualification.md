# Native MLX GRPO: bounded M4 mechanics qualification

Observed 2026-10-27 on exact execution source
`ad17275a746c045f7777336b245469f29a8fbd93`. This records one successful
restricted engine/API attempt, following an earlier retained failure. It does
not enable ordinary native admission, public logs/cancel/reconcile, arbitrary
models or rewards, environment-backed RL, or full desktop distribution.

## Exact selection and execution

The host used `TrainingAPI`/`RunsAPI`, the existing native provider family and
owned guardian, and a fresh independently reviewed, externally signed one-use
selection. Source review R7 accepted the exact execution commit before start.
The original `19768026` attempt remains consumed and unclassified at exit 122;
its authority was not renewed or replayed.

- Hardware: Apple M4, 24 GiB, Darwin arm64, macOS 27.0.1 / build 26A434.
- Separate private CPython 3.11.14 / MLX 0.32.2 / MLX-LM 0.32.2 runtime,
  with the unchanged 54-package lock
  `7c167018ee6bd61a3609971e3f834082c798e7bd2a3fa0c5c2564887da4fde4f`.
- Model: `mlx-community/Qwen3-0.6B-4bit`, revision
  `73e3e38d981303bc594367cd910ea6eb48349da8`.
- Capture: `795a56a235f549d68d67280f6576342f9010839cd09772a182531097346987b4`;
  all 744 captured source members matched Git. Runtime/native-library bytes
  matched the preceding reviewed capture. The OS shared cache remains an
  explicit trust boundary.
- Selection: `170bf2bfeeee8462b75f27cc401e1112a624788d065d22a82a3f124c05a119ae`.
  Workload: `c85b2407701fd9bce79f7f0edf540b3da828c4714fd8cd5a72d9b0e8f8051e0a`.
- Run: `8821818d-0b9e-449f-a56f-794e21fd1994`;
  attempt: `fcaa7a21-b6b8-409c-87b8-0aca1b75e20e`.

The genuine prepared input contains two synthetic color-choice prompts, 118
bytes, with semantic digest
`45c53c9395bf579b59af0172ce0368c4ac2bac2a8457d97d5fd0d4a15bdaf074`.
This is a mechanics fixture, not a practical tool-use dataset or quality test.
Its case-insensitive blue-word reward reads only generated text. Thinking is
disabled in the configured chat scaffold. The actual pinned tokenizer produced
23 IDs per prompt; a separate copy with an extra empty directory was refused
before tokenizer loading/output creation.

Controls: three updates, batch/group 16, seed 17, unit temperature, beta 0.1,
64 prompt / 8 completion / 128 total tokens, rank-four q/v LoRA on the last two
layers, alpha 32, dropout zero and constant AdamW at 1e-4. The admitted bounds
were 384 generated tokens, 48 reward calls, 64 KiB rollout evidence, 300 seconds
plus five-second termination grace, 1 MiB per output stream, 64 MiB per artifact
and 128 MiB aggregate. The host owns each exact selection and authority.

## Retained results

The owned worker exited zero with termination confirmed. Public observation
reported success; `runs.verify` returned true at
`2026-10-27T15:59:00.676755Z`. All five roles downloaded through the public
stream, and operator readback matched the Mac hashes.

| Check | Observed result |
| --- | --- |
| Actual optimizer updates | 3 |
| Complete rollout groups / reward calls | 3 / 48 |
| Groups with float32 reward variation | 3 |
| Generated tokens | 102 |
| Saved adapter parameters changed | 40,960 / 40,960 |
| Recomputed adapter delta L2 | 0.0407861391343181 |
| Saved-adapter reload replay objective | -0.013764187693595886 |

The reload value is a finite GRPO objective on retained sampled rollouts,
computed after loading the saved adapter into a fresh base. It is a consistency
check, not held-out validation or evidence of better color choice/tool use.
The operator separately decoded the initial/final safetensors, recomputed every
parameter difference, checked rewards/group counts and authenticated the
delivered workload and prompt IDs. No model execution was used for readback.

| Public role | SHA-256 |
| --- | --- |
| final_model | `41fcc255b3fdc48f0e1db9f125dee41ea6c90fb0ae4a854994209589f7813b72` |
| tokenizer | `18c0ef285ab5aeb9599a2b620e190882ffecd0282853fa24c4cba9b4d41ca6d2` |
| training_lineage | `0d6191e844a1594a2c374aff96bd13269f1d4299b4acdde60adedf49ac849790` |
| training_metrics | `03d8095e12a689d5e19a00b60c298f081e65ce45bba933b1fd608c2b19b9debb` |
| workload_record | `755f8f04ca7de2ad8045095914b02917e2ad539f10baf2d44fdce0d86ff864be` |

The final adapter SHA-256 is
`79e665eed6991efcb6f339be01134fa8d30e481f20a08270528627af6fcd570c`.
Private admissions, captures, original failure and delivered evidence remain
retained under their original run/source identities.

## Integration scope

See the [objective and supported controls](../architecture/native-mlx-grpo.md),
[host contract](../architecture/native-mlx-host.md) and
[config-first preparation example](../../examples/training_qualification/native_grpo_preparation/README.md).
This uses a separate named GRPO profile and private runtime; it does not modify
or relabel the qualified SFT environment. A new model, workload or reward needs
its own evidence and admission. One mechanics success does not establish
repeatability, training quality, cancellation/caller-loss/restart recovery or
broad stable-v1 native support.
