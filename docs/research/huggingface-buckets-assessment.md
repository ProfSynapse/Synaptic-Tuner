# HuggingFace Buckets — Integration Assessment

**Date**: 2026-03-11
**Status**: Research / Recommendation
**Feature**: [HuggingFace Storage Buckets](https://huggingface.co/docs/huggingface_hub/en/guides/buckets)

---

## What Are HF Buckets?

HuggingFace Buckets are **S3-like mutable object storage** on the Hub, powered by the Xet storage backend. Unlike HF repositories (git-based, version-tracked), buckets are **non-versioned remote object storage containers** with content-addressable deduplication.

### Key Characteristics

| Property | HF Repos (current) | HF Buckets (new) |
|----------|-------------------|-------------------|
| **Backend** | Git + LFS (now Xet) | Xet object storage |
| **Versioning** | Full git history | None |
| **Mutability** | Commit-based | Direct overwrite |
| **Deduplication** | Per-repo | Cross-bucket, byte-level chunking |
| **Best for** | Final models, datasets | Checkpoints, logs, intermediate artifacts |
| **Discoverability** | Hub search, model cards | Browsable but not searchable |

### Pricing

- **$12/TB/month** (public), **$8/TB/month** at 500 TB+
- Egress and CDN included (no transfer fees)
- Undercuts AWS S3 Standard (~$23/TB/month) by ~3x at scale

### SDK Requirements

- `huggingface_hub >= 1.5.0`
- Python API: `create_bucket`, `sync_bucket`, `batch_bucket_files`, `list_bucket_tree`, `download_bucket_files`
- CLI: `hf buckets create|sync|cp|list|rm|delete|move`

### Killer Feature: Content Deduplication

Xet uses content-defined chunking. When you retrain a model and only 5% of weights change, **only that 5% is re-uploaded**. This is transformative for iterative training workflows where checkpoints share most of their content.

---

## Current Repo Storage Architecture

The Synaptic-Tuner pipeline currently uses:

1. **Local filesystem** for everything during training (checkpoints, logs, final models)
2. **HuggingFace Repos** (git-based) for final model upload via `shared/upload/` framework
3. **No cloud/S3 storage** whatsoever

### Current Upload Flow

```
Training → local checkpoints/ → local final_model/
    ↓
UploadOrchestrator (shared/upload/orchestrator.py)
    ├── SaveStrategy (lora | merged_16bit | merged_4bit)
    ├── HfApi.upload_folder() → HF git repo
    ├── GGUF conversion → HfApi.upload_files()
    └── Model card + manifest → HfApi.upload_file()
```

### Pain Points Buckets Could Address

| Pain Point | Current State | With Buckets |
|------------|--------------|--------------|
| **Checkpoint persistence** | Local only; lost if disk fails or VM terminates | Synced to remote bucket continuously |
| **Cross-machine training** | Must manually copy checkpoints | `hf buckets sync` to resume anywhere |
| **Training logs** | Local JSONL files | Streamed to bucket for remote monitoring |
| **Intermediate artifacts** | Accumulate on local disk | Offloaded to cheap remote storage |
| **Iterative model versions** | Each upload is full-size to git repo | Dedup means only diffs are uploaded |

---

## Integration Opportunities (Ranked by Value)

### 1. Checkpoint Syncing (HIGH VALUE)

**What**: Automatically sync training checkpoints to a bucket during/after training.

**Why it matters**:
- Training runs on RTX 3090 can take hours; a crash means restarting from scratch
- Checkpoints are large (7B model = ~14 GB per checkpoint) and temporary — poor fit for git repos
- Xet deduplication means successive checkpoints (which share 95%+ content) upload fast
- Enables resuming training from any machine

**Integration point**: `Trainers/rtx3090_sft/train_sft.py` and `Trainers/rtx3090_kto/train_kto.py` — add a callback or post-checkpoint hook that runs `sync_bucket()`.

**Effort**: Low-Medium. A training callback + config option.

### 2. Training Logs & Metrics (MEDIUM VALUE)

**What**: Stream training logs/metrics to a bucket for remote monitoring.

**Why it matters**:
- Currently logs are local JSONL only
- Can't monitor training remotely without SSH
- Lightweight files, minimal bandwidth

**Integration point**: Logging callbacks in trainers.

**Effort**: Low. Periodic `batch_bucket_files()` call with log bytes.

### 3. Dataset Staging Area (MEDIUM VALUE)

**What**: Use buckets as a staging area for synthetic data generation before final dataset curation.

**Why it matters**:
- SynthChat generates data iteratively; intermediate outputs don't need git history
- Multiple generation runs can share a bucket
- Easy to share raw data across team members

**Integration point**: `SynthChat/run_generation.py` output path option.

**Effort**: Low. Add `--bucket` flag to generation scripts.

### 4. Upload Pipeline Enhancement (LOW-MEDIUM VALUE)

**What**: Add bucket as an alternative upload target alongside git repos.

**Why it matters**:
- GGUF files are large and don't benefit from git versioning
- Intermediate model formats during conversion could be stored in buckets
- Dedup across model versions saves storage costs

**Integration point**: `shared/upload/uploaders/` — add `BucketUploader` alongside existing `HuggingFaceUploader`.

**Effort**: Medium. New uploader class + registry entry + CLI flag.

### 5. Evaluation Artifacts (LOW VALUE)

**What**: Store evaluation results, comparison reports in a bucket.

**Why it matters**: Minor convenience for sharing results.

**Effort**: Low but low impact.

---

## Recommendation

### Worth integrating? **Yes, selectively.**

The highest-value integration is **checkpoint syncing** (#1). It solves a real problem (training resilience), leverages the killer feature (deduplication makes successive checkpoint uploads cheap), and fits naturally into the existing training pipeline via callbacks.

### Suggested Approach

**Phase 1 — Checkpoint Syncing** (start here):
- Add optional `--bucket` flag to training scripts
- Implement a `BucketCheckpointCallback` that syncs after each checkpoint save
- Use `sync_bucket()` with `--delete` to keep only the N most recent checkpoints remotely
- Graceful degradation: if bucket isn't configured, training works exactly as today

**Phase 2 — Training Logs** (easy add-on):
- Extend the callback to also sync log files
- Or use `hf buckets cp -` to pipe metrics directly

**Phase 3 — Evaluate broader adoption** based on Phase 1 experience.

### What NOT to change

- **Final model uploads should stay on git repos** — they benefit from versioning, model cards, Hub discoverability, and the existing upload orchestrator works well
- **Datasets should stay local/git** — they're curated artifacts that benefit from version control
- Don't add buckets as a replacement for git repos; they're complementary

### Prerequisites

- Upgrade `huggingface_hub` to >= 1.5.0
- Ensure `HF_TOKEN` has bucket permissions (likely same write token)
- Add bucket configuration to training configs (bucket_id, sync frequency, retention policy)

---

## Key Sources

- [HF Blog: Introducing Storage Buckets](https://huggingface.co/blog/storage-buckets)
- [HF Docs: Buckets Guide](https://huggingface.co/docs/huggingface_hub/en/guides/buckets)
- [HF Changelog: Introducing Buckets](https://huggingface.co/changelog/introducing-storage-buckets)
- [HF Storage Pricing](https://huggingface.co/storage)
- [GitHub Issue: Buckets API (#3796)](https://github.com/huggingface/huggingface_hub/issues/3796)
