# Research: higgsfield-ai/higgsfield — integrate, borrow, or skip?

**Date**: 2026-09-21
**Repo assessed**: https://github.com/higgsfield-ai/higgsfield @ `d2a0a05` (last commit 2024-02-14)
**Context**: Evaluating whether the "fault-tolerant GPU orchestration + ML framework" project is
worth integrating into Synaptic Tuner, or worth borrowing ideas from, while the public API
facade v1 phase is in flight.

---

## Verdict

**Do not integrate. Do not vendor. Borrow exactly one idea, and reimplement it.**

The project is abandoned, structurally broken at its launch path, architecturally aimed at a
deployment model Synaptic Tuner deliberately does not use, and its code conflicts with three
standing rules in `CLAUDE.md` (secret handling, no hardcoded scenario behaviour, no rot).

The one transferable idea is the declaration-to-workflow codegen described in section 5. It is
roughly half a day of original work and should not be copied from this source.

---

## 1. Project state

| Signal | Value |
|---|---|
| Last commit | 2024-02-14 (`Update llama.py`) |
| Total commits | 44 |
| Python LOC | ~3,050 total; ~1,960 excluding the vendored `rl/rl_adventure_2` notebooks |
| PyPI releases | `0.0.2` (2023-10-13), `0.0.3` (2023-10-16), `0.0.4rc0` (2024-03-23) |
| License | Apache-2.0 |
| Stars | ~5.6k |
| Open issues | Spam and one-word titles dating 2025-2026; no maintainer replies since 2023. `#37` ("Some issues if try to use it in a real life :)", Nov 2023) is still open and unanswered. |

The star count is a Nov-2023 launch artifact and is not evidence of maintenance. The `higgsfield-ai`
org has since become a consumer AI-video company; this repository is residue from the pre-pivot
infrastructure product.

## 2. The launch path is broken at the source

The generated GitHub Actions workflow (`higgsfield/static/templates/experiment_action.j2`) and the
node bootstrap (`higgsfield/internal/ci/setup.py:15-19`) both install a Go binary called `invoker`:

```
wget https://github.com/ml-doom/invoker/releases/download/latest/invoker-latest-linux-amd64.tar.gz
```

`github.com/ml-doom/invoker` is no longer publicly available (page returns 404; anonymous
`git ls-remote` prompts for credentials). `invoker` is the component that actually performs the
multi-node launch, the `--max_repeats` retry loop, and the run/kill lifecycle — that is, the
"fault tolerance" in the project's own description. The Python package is a thin CLI and codegen
layer around a closed binary that has been withdrawn.

Consequence: the advertised capability cannot be reproduced today from this repository at all.

## 3. Architecture mismatch

higgsfield's execution model:

```
git push main
  -> GitHub Actions (deploy.yml)   -> appleboy/ssh-action -> git pull on each GPU box
  -> GitHub Actions (<exp>.yml)    -> appleboy/ssh-action -> invoker -> torchrun -> FSDP
```

It assumes **persistent, self-owned bare-metal GPU nodes with static hostnames**, addressed over
SSH, with the repo checked out at `~/higgsfield/<project>` on every node, and GitHub Actions as the
control plane (`concurrency.group: main` is the entire queueing mechanism).

Synaptic Tuner's model is the opposite: ephemeral managed compute reached through provider adapters,
with the host owning credentials and mutable state. HF Jobs submits via
`huggingface_hub.run_job(image=..., command=["bash","-c",...], flavor=...)` and polls `inspect_job()`
(`tuner/backends/training/cloud/hf_jobs_backend.py:83-92`); Modal launches through
`tuner/execution/providers/modal/coordinator_launch.py:177-204` behind signed command bindings.
Artifacts return by bucket sync (`shared/cloud_artifacts.py:239,273,373`), not by reading a
directory on a box we own.

None of higgsfield's orchestration transfers, because the surface it targets — a machine we keep
running, with a hostname, that we can SSH into — does not exist in this design.

## 4. Direct conflicts with `CLAUDE.md`

**Secrets.** Rule: *"Secrets are referenced by name only and resolved at execution time. Never put
credential values in code, configs, logs, argv or commit messages."*

- `internal/ci/setup.py:56-63` writes the SSH private key to `~/.ssh/temp-<project>.key`.
- `internal/experiment/builder.py:69-71` codegens workflow lines that `echo` each GitHub secret's
  **value** into a plaintext `env` file on every remote host.
- `internal/ci/setup.py:13` installs Docker by `curl`-ing a personal gist and piping it to bash.

**No hardcoded scenario behaviour.** Rule: *"Tool-call formats, workspace structures and label
mappings come from config."*

- `dataset/openai.py:4-12` hardcodes an `###ASSISTANT:` prompt template.
- `llama/llama.py:113-118` hardcodes the FSDP wrap policy to `LlamaDecoderLayer`.

**Import-time side effects.** `loaders/llama_loader.py:32` uses
`tokenizer=LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")` as a *default argument* —
evaluated at import, so importing the module downloads a gated model.

**Rot.** `optimum.bettertransformer` (`llama/llama.py:32`) is deprecated in favour of native
`torch.nn.functional.scaled_dot_product_attention` in Transformers and is slated for removal in
Optimum v2. `LlamaTokenizer` is the slow tokenizer. `pyproject.toml` pins `python = "^3.8.13"`.
Model coverage is Llama-2 and Mistral only.

## 5. The one idea worth taking

`internal/experiment/{decorator,ast_parser,params,builder}.py` implements:

1. A declarative `@experiment("name")` / `@param("size", options=[...], default=..., description=...)`
   decorator pair on the training entry point.
2. An **AST parser** that reads those declarations out of the source *without importing the module*
   (`ast_parser.py`) — so workflow generation never has to import torch.
3. A codegen step (`params.py:as_github_action`, `builder.py`) that emits a
   `workflow_dispatch` GitHub Action with typed inputs: `type: choice` with `options` for
   enumerated params, `type: boolean` for bools, plus `required`/`default`/`description`.

The result is a click-to-launch form in the GitHub UI, regenerated automatically on every push to
`main`, that stays in sync with the training code's declared parameters.

**Why it is interesting here.** Synaptic Tuner already has the declarative half twice over:
experiment specs under `Trainers/cloud/experiments/*.yaml` driving
`python tuner.py run-experiment --experiment-spec ...`, and recipe YAMLs under
`Trainers/recipes/*.yaml` driving `local-run` / `cloud-run --job-config ...`. Generating one
`workflow_dispatch` action per checked-in spec would give a no-CLI launch surface for non-agent
operators and a typed, validated parameter form for free.

**Why we should not copy the code.** Our specs are already YAML, so the AST trick — the genuinely
clever part — is unnecessary; we would read the spec files directly. What remains is Jinja
rendering of a workflow template from a typed parameter list, which is small, and which we would
want wired to our own secret-by-name discipline rather than higgsfield's `echo $SECRET >> env`.

Estimated cost if pursued: ~half a day. Not urgent, and orthogonal to the facade phase.

## 6. The real gap it gestures at (and does not fill)

higgsfield's only substantive ML content is FSDP/ZeRO-3 sharding for models that do not fit one GPU
(`llama/llama.py`, 301 lines; `checkpoint/fsdp_checkpoint.py`, 108 lines).

Synaptic Tuner has **no** multi-node or sharded training. A repo-wide search for `fsdp`,
`deepspeed`, `torchrun` and `WORLD_SIZE` returns no hits in training code across all six methods
(sft, kto, dpo, grpo, embedding, ace_step); `accelerate` appears only as a transitive dependency.
The constraint is explicit, not accidental:

- `tuner/cloud/hardware_planner.py:431-442` — `_stage_supports_flavor()` returns
  `(False, "multi-GPU training is not wired for this stage yet")` for any flavor with `gpus > 1` at
  the `training` stage. Multi-GPU is permitted for `loss`, and for `evaluation` on the vLLM runtime.
- `tuner/backends/training/cloud/_hf_command_builder.py:229-244` builds a bare
  `python train_<method>.py ...` — one process, one GPU.
- Fault tolerance is bucket sync (`shared/cloud_artifacts.py:386 HFBucketSyncCallback`) plus manual
  `--resume-from-checkpoint`. No elastic rendezvous, no preemption recovery.

That is coherent given the Unsloth-based trainer stack, which is single-GPU in its open release.
The scaling strategy here is cheap LoRA on one GPU plus many sequential experiments
(`shared/flywheel/experiment_loop.py`), not scaling one run.

If we ever cross that line, the candidates are Accelerate/TRL FSDP configs, torchtune, or Axolotl —
all maintained, all model-general. higgsfield's implementation is Llama-2-shaped, two and a half
years stale, and depends on a deprecated Optimum API.

Note also where such a framework could legally live. `pyproject.toml` keeps the engine package
ML-dependency-free (`PyYAML`, `jsonschema`, `packaging`, `python-dotenv`, `requests`), and
`docs/architecture/api-facade-v1.md:52-58` gates that with an import-closure test: after importing
`synaptic_tuner.api.v1`, `huggingface_hub`, `modal`, `runpod`, `sqlite3` and `tuner` must all be
absent from `sys.modules`. Any distributed launcher therefore belongs on the trainer side of the
boundary — inside the pinned image and the recipe — never in `synaptic_tuner/`. higgsfield is built
the other way round: its package *is* the launcher.

**Open question for the maintainer:** is single-GPU training an active ceiling, or is the ephemeral
single-GPU HF Jobs / Modal path still sufficient for the target model sizes? The answer decides
whether the section-6 gap is worth any work at all. It does not change the verdict on higgsfield.

## 7. Summary table

| Dimension | higgsfield | Synaptic Tuner today |
|---|---|---|
| Maintenance | Dead since 2024-02 | Active |
| Compute model | Self-owned persistent SSH nodes | Ephemeral HF Jobs / Modal |
| Control plane | GitHub Actions + missing Go binary | `tuner.py` CLI + provider adapters |
| Training methods | Causal LM full-finetune (Llama-2, Mistral) | SFT, KTO, DPO, GRPO, embedding, ACE-Step, classical ML, MLX |
| Distributed | FSDP / ZeRO-3 (Llama-shaped, stale) | None (deliberate, single-GPU) |
| Data generation | None | SynthChat |
| Evaluation | None | Evaluator, assertion-driven YAML |
| Secret handling | Values echoed into remote plaintext files | Referenced by name, resolved at execution |
| Public API | None | `synaptic_tuner/api/v1` ports/adapters + import-closure gates |
| Reproducibility | `pip install higgsfield==0.0.3` in a workflow | Pinned image digests, hash-required installs, `modal-runtime-v1.lock.json` |

---

## Sources

- https://github.com/higgsfield-ai/higgsfield (clone inspected directly)
- https://pypi.org/pypi/higgsfield/json
- https://github.com/higgsfield-ai/higgsfield/issues
- https://github.com/ml-doom/invoker (404 / not publicly readable)
- https://github.com/huggingface/optimum/issues/2083 (BetterTransformer deprecation)
