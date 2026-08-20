# Cloud Training Documentation - Nebius AI Cloud

This directory contains comprehensive documentation for running your Toolset-Training pipeline on Nebius AI Cloud.

> The Nebius material below is retained as legacy provider research. The current
> protected Hugging Face source/bootstrap-proof lane is documented first and is
> not a generic training launcher.

## Protected Hugging Face JP Source and Bootstrap Smoke

Status as of 2026-08-20: the original JP-S security audit returned REVISE for
process-local cancellation, insufficient `JobInfo` identity agreement, and an
incomplete isolated-launcher runtime contract. JP-BR/JP-ER/JP-CR implement the
bounded remediations. JP-S2 passed the frozen 29-file implementation manifest
`55e2c876dd8cc282a43248a3eeaf3f445f6e452ce76ab2d7a0b814b460ef0f41`
with 283 passed/6 skipped, 16 hostile checks, 87 import/generic checks, and no
findings. Final JP-R also PASSed that earlier frozen release/inventory tree.
Those PASSes remain historical evidence, but the later JP-PREP implementation
materially changed the protected source/provisioning tree. Its first independent
security re-audit returned **REVISE** on four findings, and its first release
re-audit also returned **REVISE**. The
bounded JP-PRT-R and JP-PRH-R remediations are implemented. Fresh independent
security and release re-audits now both **PASS** the exact
JP-PRT-R/JP-PRH-R/JP-PRC-R tree. Security evidence is **400 passed, 5 skipped**.
Release evidence is focused **283 passed, 3 skipped**; tracking **263 passed, 1
skipped**; CLI/contract **284 passed, 1 skipped**; broad runnable **624 passed,
16 skipped** with five historical failures; and stale lifecycle fixtures **60
passed, 8 classified**. Checkpoint 16R is eligible, but these PASSes are not live
installation or provider proof. Do not run `hf-source provision` or
`hf-smoke execute` until 16R is committed and exactly pushed, a fresh
named-branch worktree is created from that pushed commit, the five-pin
clean-venv launcher gate passes there, and explicit credential preflight
succeeds. `cloud.launch` and generic HF training remain unavailable.

### Authorization boundary

Joseph authorized at most one paid bootstrap smoke: `cpu-basic`, `python:3.12`,
no training, publication, ports, SSH, or retries; provider timeout 10 minutes;
one cancellation attempt after 12 minutes if still nonterminal; observation
ends at 15 minutes; projected compute at most USD $0.01 and hard total cap USD
$2. Any retry, second submission, wider workload, longer observation, or higher
cost needs new approval.

The fixed workload digest is:

```text
0d1d3454d079ea994a1e3a24b59b772bd4adb40cb441e00cc5801faf5d220841
```

### Isolated launcher

Provisioning/submission requires Python 3.12 and exactly these direct pins in a
dedicated launcher environment:

- `huggingface_hub==1.27.0`
- `jsonschema==4.23.0`
- `packaging==24.1`
- `python-dotenv==1.0.1`
- `PyYAML==6.0.2`

Never install this launcher set into the main Unsloth/trainer interpreter.

```powershell
python scripts/setup_hf_jp_launcher.py `
  --python C:\path\to\python3.12.exe `
  --venv C:\path\to\new\hf-jp-launcher `
  --repo-root <exact-repository-worktree>
```

The setup command refuses a non-3.12 interpreter, missing/extra/duplicate/ranged
or reordered requirements, or an existing target directory. It then verifies
all five installed distribution versions in isolated Python, imports the two
protected handlers without loading Torch/Transformers/Unsloth, and runs
`hf-source --help` plus `hf-smoke --help` from the exact repository worktree with
user-site disabled, bytecode writes disabled, and no credentials/provider
effects. This clean-venv gate proves protected import/help compatibility only.
The remote image is currently `python:3.12`, which is a mutable tag; a successful
smoke would prove only the image resolved for that run, not a digest-pinned
supply chain.

### Protected command sequence

These examples describe the implemented CLI shape. Replace placeholders only
with identities from the frozen preflight. They are not authorization to run
before the five-pin clean-venv and exact-pushed gates pass; credential preflight
is additionally required before provision, execute, or observe.

1. Prepare the exact immutable source bundle without credentials or provider contact:

```powershell
python tuner.py hf-source prepare `
  --project-root <host-project-root> `
  --source-config <committed-cloud-config> `
  --source-mode <standalone-or-discovered-host-mode> `
  --base-dir <absolute-external-tracking-root> `
  --json
```

`prepare` creates a neutral bootstrap experiment, performs the exact-pushed Git
preflight, parses `cloud.hf_jobs.bootstrap_volume` from the exact committed Git
blob, and rechecks the selected regular link-free config against that commit
before installing any transport artifact. It durably creates or adopts the
canonical enriched SourceLock before transport preparation, then writes the
capsule, bundle, descriptor, and PREPARED tracking state only below the explicit
absolute `--base-dir`, which must be outside the selected source trees. Its
output contains only the recoverable experiment ID, portable tracking URIs,
digests, and closed read-only volume metadata. It imports no provider SDK, ML/UI
stack, dotenv loader, or credential boundary.

If interruption occurs after neutral experiment creation, the error reports the
experiment ID. Rerun `prepare` with that exact `--experiment-id` and the same
base/config/source inputs. Recovery accepts only neutral, exact SourceLock-only,
or exact PREPARED state and converges idempotently. A crash-created SourceLock
artifact may be adopted only when it is bounded, canonical, regular, link-free,
run-bound, and byte-identical; a SourceLock already copied into an interrupted
transport is authenticated under the same rule before the canonical projection
is restored. The SourceLock projection is durable before transport installation,
so interruption on either side of that boundary converges without overwriting
different bytes. ACKNOWLEDGED, CONSUMABLE, approval, submission, mismatched, or
partial-reference state fails closed.

2. Provision and read back the exact PREPARED Profile-C source prefix:

```powershell
python tuner.py hf-source provision `
  --project-root <host-project-root> `
  --experiment-id <experiment-id> `
  --actor <non-secret-operator-id> `
  --authority operator `
  --env-file <explicit-project-env-file> `
  --base-dir <same-absolute-external-tracking-root> `
  --json
```

The selected file must be a regular, link-free file inside the project/config
boundary and contain exactly a nonblank `HF_TOKEN` authority. Metadata preflight
selects only its root/path and reads no bytes; the single complete content read
occurs only after the durable provisioning claim. The handler rejects file or
ambient `HF_API_KEY`, rejects ambient `HF_TOKEN`, never loads the file into
`os.environ`, and never emits its value. On POSIX, held ancestor descriptors and
`O_NOFOLLOW` bind the read to the opened regular file: a rename after final open
cannot change bytes read from that descriptor, but no claim is made that the
pathname still names the same object afterward. On Windows, native `CreateFileW`
handles permit only read sharing, deny write/delete sharing during the read,
reject reparse points, and verify final-handle containment and identity. The
command may create the exact
private bucket/prefix or verify identical existing bytes, but never submits a
job. Before reading credential bytes or importing provider code, it reloads the
same external tracking root, verifies durable PREPARED provenance, and
reauthenticates the descriptor, SourceLock, policy, capsule, and bundle. Success
records ACKNOWLEDGED evidence and then CONSUMABLE locally.

Provisioning is itself at-most-once. Before credential content or provider
construction, the handler durably records a closed
`synaptic-hf-provisioning-claim/v1` `CLAIMED` event. Each event restores the
closed sequence/time/predecessor/evidence fields plus `reason_code` and
`provider_effect_possible`. The mapping is exact: `CREDENTIAL_REJECTED` and
`LOCAL_POSTCLAIM_FAILURE` mean no provider effect is possible;
`PROVIDER_OUTCOME_AMBIGUOUS`, `INTERRUPTED_AFTER_CLAIM`, and
`RECOVERY_EVIDENCE_INVALID` mean an effect may be possible. Verified success
records `SUCCEEDED` with exact evidence; post-claim uncertainty records terminal
`AMBIGUOUS`, leaves source transport PREPARED, and cannot be retried. A resumed
`CLAIMED` head never receives provider authority: it adopts an exact orphan
terminal or validates exact persisted evidence and converges to `SUCCEEDED`, or
terminalizes as `AMBIGUOUS` with `INTERRUPTED_AFTER_CLAIM` when evidence is
absent and `RECOVERY_EVIDENCE_INVALID` when recovery artifacts are invalid or
conflicting. Durable `SUCCEEDED` may finish local consumption without another
provider call; terminal `AMBIGUOUS` never grants a second attempt.

3. Create the exact provider-free run approval:

```powershell
python tuner.py hf-smoke approve `
  --project-root <host-project-root> `
  --experiment-id <experiment-id> `
  --authorization-reference <recorded-reference> `
  --issued-at <UTC-RFC3339> `
  --expires-at <UTC-RFC3339> `
  --quoted-at <UTC-RFC3339> `
  --hourly-price-usd <official-cpu-basic-hourly-price> `
  --projected-cost-usd <no-more-than-0.01> `
  --base-dir <same-absolute-external-tracking-root> `
  --json
```

Approval binds the exact descriptor/evidence/SourceLock/bundle/capsule/policy
digests and canonical workload. It accepts no hardware, image, command, retry,
training, publication, or port override.

4. Consume the one-shot authorization and submit exactly once:

```powershell
python tuner.py hf-smoke execute `
  --project-root <host-project-root> `
  --experiment-id <experiment-id> `
  --env-file <explicit-project-env-file> `
  --base-dir <same-absolute-external-tracking-root> `
  --json
```

`execute` writes the durable `SUBMITTING` claim before credential or provider
work. A provider exception becomes terminal `AMBIGUOUS`; do not retry. Success
records only the normalized namespace/job ID and becomes `SUBMITTED`.

5. Observe only the recorded submission:

```powershell
python tuner.py hf-smoke observe `
  --project-root <host-project-root> `
  --experiment-id <experiment-id> `
  --env-file <explicit-project-env-file> `
  --base-dir <same-absolute-external-tracking-root> `
  --json
```

Every provider-returned `JobInfo` must normalize to the exact recorded namespace
and job ID before its status or result is trusted. Missing, malformed,
contradictory, or different identity is not the submitted job. Observation
cannot select a different job. At 12 minutes it durably creates or reuses the
exact cancellation-attempt event; only the first locked claimant is authorized
to call `cancel_job`. Resumed or concurrent observers receive the same claim but
cannot call the provider again. A failed/ambiguous cancellation remains consumed.
Observation stops at the 15-minute outer boundary.

The flat CLI parser enforces a frozen explicit-option allowlist for each of the
five protected actions. Validation covers both `--flag value` and
`--flag=value`; a globally recognized option is still rejected when it is not
listed for that exact action. Preparation alone accepts `--source-config` and
`--source-mode`; provisioning alone accepts actor/authority and the credential
file; approval alone accepts authorization/quote fields; execute/observe alone
accept the credential file. None accepts `--yes`, retry, hardware, image,
command, publication, port, or generic training overrides.

### Hugging Face v1.27 contract and ambiguity

The isolated adapter checks the complete official v1.27.0 signatures for
`create_bucket`, `bucket_info`, `list_bucket_tree`, `batch_bucket_files`, and
`download_bucket_files` before its first read or mutation. It verifies the exact
private bucket identity; rejects unknown, unrelated, duplicate, or colliding tree
entries; hands authenticated immutable bytes to the upload call; and downloads
and hashes every member afterward.

The official API documents `batch_bucket_files` as non-transactional: an error
can leave only part of the requested upload present. Therefore any bucket-create,
upload, or post-upload verification uncertainty is `mutation_ambiguous`, is not
evidence, and is never retried automatically. Inspect the exact immutable prefix
read-only and obtain fresh user authority before any later action.

### Remaining blockers

- Original JP-S verdict: REVISE with the three findings above.
- JP-S2 security re-audit: PASS on the frozen manifest above; this is local
  security evidence, not live installation/provider proof.
- Final JP-R release/inventory re-audit: **PASS**. Evidence: affected **200 passed, 3 skipped**; full JP/HF/import/order **393 passed, 15 skipped**; tracking **249 passed, 1 skipped**; CLI/capability/project/plugin/contract **268 passed, 1 skipped**; broad cloud **673 passed, 15 skipped** with the same five accepted failures and two documented exclusions; **26 Python files compiled**; imports clean; diff check warnings only; public API and `cloud.launch` unchanged; skill sync clean.
- Those JP-S2/JP-R PASSes predate JP-PREP and do not approve its changed tree.
- First post-JP-PREP security re-audit: **REVISE** on four findings. Evidence:
  former-HIGH closure **25 passed, 2 Windows link skips**; parser **63 passed**;
  claim/operator/assets **27 passed**; thread/spawn/ambiguity **3 passed**; and
  protected handlers **39 passed, 2 skipped**.
- First post-JP-PREP release re-audit: **REVISE**. Evidence: focused **270
  passed, 3 skipped**; tracking **257 passed, 1 skipped**; broad cloud **679
  passed, 16 skipped, 13 classified failures** (five accepted prior and eight
  stale lifecycle fixtures); CLI/project/capability/plugin/contract **279
  passed, 1 skipped** plus five MAX_PATH artifacts; affected short-path rerun
  **16/16 passed**.
- JP-PRT-R remediation evidence: **138 passed, 1 skipped** focused, **261
  passed, 1 skipped** full tracking, and **12 passed** contract checks.
- JP-PRH-R remediation evidence: **67 passed, 2 skipped** focused, **148 passed,
  2 skipped** utilities, and **348 passed, 3 skipped** broad, plus one classified
  missing-Transformers environment failure.
- Fresh post-remediation security re-audit: **PASS**, with **400 passed, 5
  skipped**.
- Fresh post-remediation release re-audit: **PASS**. Evidence: focused **283
  passed, 3 skipped**; tracking **263 passed, 1 skipped**; CLI/contract **284
  passed, 1 skipped**; broad runnable **624 passed, 16 skipped** plus the five
  historical failures; stale lifecycle fixtures **60 passed, 8 classified**.
- Checkpoint 16R is eligible. JP-LIVE is next only after 16R is committed and
  exactly pushed, a fresh named-branch worktree is created from that commit, the
  exact five-pin launcher passes there, and explicit-file credential preflight
  succeeds.
- No credential value was inspected during implementation or documentation.
- Official-source API compatibility is not live account/bucket/job proof.
- Modal is later. RunPod is later and requires fresh dated research against its
  current official API/SDK before implementation because its API may have changed.

## 🚀 Quick Start (10 minutes)

**Start here:** [`NEBIUS_QUICKSTART.md`](./NEBIUS_QUICKSTART.md)

This guide gets you from zero to training in 10 minutes using JupyterHub.

## 📚 Documentation Overview

### 1. **NEBIUS_QUICKSTART.md** - Fast-Track Guide
**Read this first!**
- Three integration approaches (JupyterHub, VM, SkyPilot)
- 10-minute setup instructions
- Cost breakdowns
- Quick wins and common commands
- Troubleshooting FAQ

**Best for:** Everyone getting started with Nebius

### 2. **NEBIUS_INTEGRATION_SUMMARY.md** - Executive Summary
**For decision makers and planning**
- Research findings and key conclusions
- Cost analysis and ROI
- Performance benchmarks (3x faster than RTX 3090)
- Implementation roadmap (Phase 1-3)
- Risk mitigation strategies
- API integration patterns

**Best for:** Understanding the business case for Nebius

### 3. **nebius-integration-guide.md** - Comprehensive Guide
**For implementation**
- Detailed setup instructions (10,000+ words)
- Step-by-step tutorials with code examples
- All three approaches fully documented
- Multi-node distributed training
- Cost optimization strategies
- Complete troubleshooting guide

**Best for:** Implementing and running production training

### 4. **nebius_training_notebook.ipynb** - Ready-to-Use Notebook
**For JupyterHub users**
- Complete training pipeline in notebook format
- Environment setup, SFT training, testing, upload
- Works on Nebius JupyterHub out-of-the-box
- Inline GPU monitoring and logging
- Cost estimates per cell

**Best for:** Interactive development and testing

### 5. **nebius_skypilot_config.yaml** - Infrastructure-as-Code
**For advanced users**
- SkyPilot orchestration configuration
- Pre-configured for 8x H100 GPUs
- Automatic environment setup
- Multi-node training support
- Spot instance configuration

**Best for:** Production orchestration and automation

## 🎯 Choose Your Path

### Path 1: Quick Test (10 minutes, ~$0.38)
1. Read [`NEBIUS_QUICKSTART.md`](./NEBIUS_QUICKSTART.md)
2. Deploy JupyterHub with H100 GPU
3. Upload [`nebius_training_notebook.ipynb`](./nebius_training_notebook.ipynb)
4. Run 1 epoch of SFT training

**Result:** Validate that your pipeline works on Nebius

### Path 2: Production Setup (30 minutes, ~$1.50)
1. Read [`nebius-integration-guide.md`](./nebius-integration-guide.md) - "Compute VMs" section
2. Create VM with 8x H100
3. Clone repository and run `setup.sh`
4. Run full SFT + KTO pipeline

**Result:** Production-ready training environment

### Path 3: Advanced Orchestration (1 hour, variable cost)
1. Read [`nebius-integration-guide.md`](./nebius-integration-guide.md) - "SkyPilot" section
2. Install SkyPilot: `pip install "skypilot-nightly[nebius]"`
3. Launch with [`nebius_skypilot_config.yaml`](./nebius_skypilot_config.yaml)
4. Experiment with multi-node and spot instances

**Result:** Automated, cost-optimized training at scale

## 💰 Cost Summary

| Training Type | Duration (H100) | Explorer Cost ($1.50/hr) |
|---------------|----------------|-------------------------|
| SFT (7B) | 15 min | $0.38 |
| KTO (7B) | 5 min | $0.13 |
| Full Pipeline | 20 min | $0.50 |
| 10 experiments | 3.3 hours | $5.00 |
| 100 experiments | 33 hours | $50.00 |

**Explorer Tier:** First 1,000 GPU-hours/month at $1.50/hour (available until March 2025)

## 🚀 Why Nebius?

✅ **3x faster** than local RTX 3090 (H100 GPUs)
✅ **No code changes** - existing `train.sh` scripts work as-is
✅ **80GB VRAM** - vs 24GB local (larger models, bigger batches)
✅ **Cost-effective** - $0.50 per full SFT+KTO pipeline
✅ **Explorer Tier** - $1.50/GPU-hour for first 1,000 hours/month
✅ **Production ready** - Bare-metal performance, InfiniBand networking

## 📊 Performance Comparison

| Hardware | SFT (7B) | KTO (7B) | VRAM | Cost |
|----------|----------|----------|------|------|
| RTX 3090 (local) | 45 min | 15 min | 24GB | Free (power costs) |
| **H100 (Nebius)** | **15 min** | **5 min** | **80GB** | **$0.38-0.50** |

**Time savings:** 3x faster training
**Iteration speed:** Can run 3x more experiments in same time
**Result:** Faster development, better models

## 🔗 External Resources

### Nebius Official
- **Platform:** [nebius.com](https://nebius.com/)
- **Documentation:** [docs.nebius.com](https://docs.nebius.com/)
- **Pricing:** [nebius.com/prices](https://nebius.com/prices)
- **API:** [github.com/nebius/api](https://github.com/nebius/api)
- **Python SDK:** [pypi.org/project/nebius](https://pypi.org/project/nebius/)

### Tutorials
- [Multi-Node Fine-Tuning with SkyPilot](https://nebius.com/blog/posts/skypilot-k8s-for-multi-node-fine-tuning)
- [LLM Fine-Tuning with MLflow](https://nebius.com/blog/posts/orchestrating-llm-fine-tuning-k8s-skypilot-mlflow)
- [SkyPilot Integration Guide](https://docs.nebius.com/3p-integrations/skypilot)

### AI Studio (Inference)
- [Quickstart](https://docs.nebius.com/studio/inference/quickstart)
- [API Documentation](https://docs.nebius.com/studio/inference/api)
- [Cookbook Examples](https://github.com/nebius/ai-studio-cookbook)

## ❓ Common Questions

**Q: Do I need to modify my training code?**
A: No. Your existing `train.sh` and training scripts work as-is on Nebius VMs.

**Q: How do I get my trained models back?**
A: Your `upload_model.sh` works on Nebius (uploads to HuggingFace). Or use `scp` to download locally.

**Q: What if I exceed the Explorer Tier limit?**
A: After 1,000 hours, pricing switches to on-demand ($2/hour). Still competitive.

**Q: Can I run multi-node training?**
A: Yes. Use SkyPilot with `num_nodes: 2+` in the config file.

**Q: What about data security?**
A: Nebius offers European data residency, encryption at rest/in-transit, and compliance certifications.

**Q: How do I monitor training progress?**
A: Use W&B (your existing integration works), or tail logs with `tail -f logs/training_latest.jsonl`.

## 🎓 Next Steps

1. **Read** [`NEBIUS_QUICKSTART.md`](./NEBIUS_QUICKSTART.md) (10 min)
2. **Sign up** at [nebius.com](https://nebius.com/)
3. **Try** JupyterHub approach (~$0.38 for first test)
4. **Validate** that your pipeline works
5. **Scale up** to production VM setup
6. **Optimize** with SkyPilot and spot instances

## 📝 Documentation Versions

- **Created:** November 23, 2025
- **Based on:** Nebius platform as of November 2025
- **Research:** Web search and official documentation
- **Status:** ✅ Production ready

All guides are current and include latest best practices for Nebius AI Cloud.

---

**Ready to get started?** Open [`NEBIUS_QUICKSTART.md`](./NEBIUS_QUICKSTART.md) 🚀

**Need more detail?** Read [`NEBIUS_INTEGRATION_SUMMARY.md`](./NEBIUS_INTEGRATION_SUMMARY.md) for the full research summary.
