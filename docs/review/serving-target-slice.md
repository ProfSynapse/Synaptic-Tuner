# Serving target slice

This internal slice converts a freshly validated `RetrievedSFTModel` into the
paths needed by a future model loader. Full-model artifacts use their retrieved
model and tokenizer directories directly and perform no base-model preparation.
LoRA artifacts request exactly the authenticated workload's model reference and
pinned revision through `PinnedModelPreparer` and require an exact typed result.

`capture_pinned_model_snapshot` is the single local snapshot capture path shared
with provider composition. It walks a canonical relative snapshot beneath a
configured private root through directory-relative, no-follow descriptors and
records a sorted inventory of regular single-link files with size, SHA-256,
device, and inode. Fresh validation repeats those checks and binds the serving
target to the retrieved model identity. Bounds are 20,000 files, 20,000
directories, depth 64, and one TiB total.

Model identity and path metadata, platform support, entry count, directory
depth, every declared file size, and the aggregate size are admitted before
any model file is hashed. This prevents a rejected oversized inventory from
causing an unbounded read first.

These checks establish local integrity, not model provenance or a new
authentication framework. The trusted preparer remains responsible for pinned
upstream preparation. The same-user host process is assumed non-hostile. There
is no SDK, cache policy, downloader, database, GPU use, model load, CLI, public
export, or persisted receipt in this slice.
