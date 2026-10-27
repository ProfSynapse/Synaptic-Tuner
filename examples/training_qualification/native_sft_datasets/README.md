# Representative native SFT dataset preparation

This host example prepares two authentic immutable conversation publications.
It fills the bounded acquisition/subset CLI gap while using the existing
conversation normalizer, verifier, shared SFT preprocessor and native batch
contract. It does not install ML packages, load model weights, import MLX,
create training authority or launch training. The `prepare-dataset` CLI only
accepts the ingestion v1/v2 configs; it cannot acquire parquet or select these
text/tool conversation subsets. No training runtime was changed for a corpus.

## Input contract and selections

- `everyday.json`: HuggingFaceTB/everyday-conversations-llama3.1-2k at immutable
  revision `14f543216b9ba42b6b951dc5bd199460d193b162`. Both parquet members have
  exact source hashes, byte lengths and row counts. Publisher README SHA-256
  `7c0b4d4c9c8de6651d3646d91e92638d46beb67f867f7200a996b72f2611d95a`
  declares Apache-2.0, 2,260 `train_sft` and 119 `test_sft` conversations.
  Select 100 train from `train_sft`, 20 validation from `test_sft`. Never
  reassign a declared source split. All source test topic groups are reserved,
  including topics of excluded unanswered conversations.
- `nexus-tools.json`: exact five canonical JSONL files from engine merge
  `3e357a2f0d1d85bdf06c67274946d7a9846ff742`, total 3,504 rows.
  Require **boolean** `label: true`; 1,053 absent labels are excluded, not
  interpreted as approval. Select 20 train and 4 validation per manager:
  contentManager/v2.3, memoryManager/v2.4, promptManager/v2.6,
  searchManager/v2.2, storageManager/v2.4. Synthetic session/workspace context
  remains complete and in order.
- Explicit `exclude_and_report` source eligibility omits conversations without
  a final assistant or with empty messages. The retained coverage lists every
  such original row number. Observed: 255 HF train, 13 HF test, and two positive
  storage rows (12 and 367). No selected row is truncated or dropped.
- Before sampling, union exact complete conversations, normalized complete user
  prompts, normalized final targets, target-as-context aliases, and configured
  source groups (`full_topic` on HF). Seeded hash ranking chooses one original
  representative per group. Validation source groups win over training aliases.
  Exact selected held-out target prose appearing in any training message rejects.
  These checks cannot establish every paraphrase/semantic relationship; source
  truth and representative semantic review remain separate requirements.
- Retain source ordering within each conversation and select its final assistant
  as target. Earlier assistant turns are masked context. Source metadata is
  preserved in the retained original files and committed through each selected
  raw-row hash, file hash, original row number, source ID and config provenance.
  Auxiliary metadata is not inserted into the assistant target.

The tool projection explicitly calls the existing generic
`sanitize_messages_for_chat_template`: null assistant content plus structured
calls becomes `tool_call: <function>\narguments: <JSON>`. It preserves all
arguments and does not rewrite commands. The selected historical `useTools`
wrapper, positional CLI strings and old system instructions do **not** match
the current `nexus use -- ... --flag value` contract in every case. This lane
tests learning historical serialized demonstrations, not modern Nexus execution,
native Qwen structured tool calling, reward quality, or tool-task correctness.
Review original calls and complete rendered targets before admission.

## Linux acquisition and immutable publication

Use a CPU-only Python environment with the pinned `pyarrow==23.0.1` parquet
reader and the engine's base dependencies. The example needs no datasets SDK,
torch, CUDA, HF token or provider client. `profile` uses the already reviewed
Mac runtime's Transformers tokenizer; do not install or change that runtime.

```sh
umask 077
stage=/ABSOLUTE/NEW/PRIVATE/sft-datasets
mkdir -m 700 "$stage"  # Its parent must already exist.
config=examples/training_qualification/native_sft_datasets

python -B "$config/prepare.py" acquire --config "$config/everyday.json" \
  --output "$stage/hf-acquisition"
python -B "$config/prepare.py" prepare --config "$config/everyday.json" \
  --acquired-root "$stage/hf-acquisition" --output "$stage/everyday"
python -B "$config/prepare.py" prepare --config "$config/nexus-tools.json" \
  --source-root "$PWD" --output "$stage/nexus-tools"
```

Every action's destination is new; files are exclusive 0600, directories 0700.
The Linux publisher retains its existing descriptor custody, no-replace publish,
fsync and independent readback. Originals are read-only bounded stable reads.
Public acquisition disables proxy inheritance and credentials, restricts HTTPS
redirect origins, limits redirects to three, reads at most declared bytes plus
one, checks exact source size/hash before retaining a source member, and has a
60-second per-source deadline with 20-second network timeouts. A failed directory
or acquired first member remains evidence; a new reviewed location is required
for another acquisition. No automatic retry, overwrite, repair or cleanup occurs.
Pin trusted parquet archives before decoding; the decoded table's 64 MiB limit
is a postdecode check, not a hard decompression-memory sandbox.

Each successful preparation retains `selected.jsonl`, `normalizer.json`,
`coverage.json`, `identity.json`, and `dataset-<digest>/{dataset.jsonl,manifest.json}`.
The identity comes from the existing verified publication, not a filename.
Both example sets have 100 train and 20 validation rows and are below the native
2 MiB executable dataset limit. Use each checked-in `*.identity.json` to compare
the reproduced semantic/content identity. Preserve old publications on change.

## Exact tokenizer profile and proposed one-pass run

The complete strict trainer `doctor.yaml` is tokenizer-only diagnosis config.
Override its model with the already reviewed local tokenizer snapshot and set
the exact selected sequence length. Run on all 120 rows:

```sh
python -B tuner.py doctor sft-mask --sft-config "$config/doctor.yaml" \
  --dataset-path /PRIVATE/PUBLICATION/dataset.jsonl \
  --model /PRIVATE/REVIEWED/tokenizer-snapshot --max-seq-length 512 \
  --sample-size 0 --preview-rows 0,20 --json
```

Use 3,072 for Nexus. Initial evidence at 2,048 passes HF (max 305); Nexus
correctly rejects nine overlength rows. The doctor's reported maximum excludes
rows that fail materialization. A later 8,192-token diagnostic passes all Nexus
rows, max 2,796; it is not the final selected profile or training evidence.
Earlier incomplete doctor YAML reports are retained; their closed TypeError was
required trainer dataclass fields missing from that config, corrected entirely
in YAML.

On the final reviewed source, inside the existing private Mac runtime, execute
the checked-in `profile` action with the original tokenizer snapshot manifest:

```sh
python -B "$config/prepare.py" profile --config "$config/everyday.token-profile.json" \
  --publication /PRIVATE/everyday/dataset-38f84d342479589e7de6220c5cc7359418543e26b67e4e3c9816bfba23a27483 \
  --tokenizer-snapshot /PRIVATE/REVIEWED/tokenizer-snapshot.json \
  --output /PRIVATE/NEW/everyday-token-profile
python -B "$config/prepare.py" profile --config "$config/nexus-tools.token-profile.json" \
  --publication /PRIVATE/nexus-tools/dataset-c6bca71a6463a8d89cf4bcac970985315b1f480e4d5f3d27050a81fb728b64c2 \
  --tokenizer-snapshot /PRIVATE/REVIEWED/tokenizer-snapshot.json \
  --output /PRIVATE/NEW/nexus-token-profile
```

This verifies the tokenizer through a retained `NativeTree` before and after
`AutoTokenizer` (`local_files_only=true`, `trust_remote_code=false`, fast tokenizer),
materializes exact shared token IDs and labels, admits them with
`prepare_mlx_sft`, and uses the actual native seeded batch iterator. No text
loader, extra EOS, random split, truncation, dropped rows or tail omission is
permitted. `tokens.jsonl` retains every row ID/split, complete IDs/labels and
generation prefix. `complete-examples.json` retains full configured specimens.
`token-profile.json` retains actual/supervised/padded tokens, batch row IDs,
tokenizer commitments, source identity and evidence digest. The explicit 16 MiB
aggregate output budget rejects before any evidence directory/write. CLI output
contains only summaries and digests, within the 1 MiB process stream limit.

The two `*.proposal.json` documents and canonical `*.training-input.json` files
propose independent fresh runs through existing `native_host.py`. HF selects
512 tokens; Nexus 3,072. Each uses batch 2, 50 updates, seed 17, last-two-layer
rank-4 q/v LoRA, constant AdamW 1e-4, report every 10, evaluate/save every 50,
all 20 held-out rows per evaluation, and a separate saved-final-adapter reload
evaluation. One traversal has 50 complete training batches and 10 validation
batches. Fifty updates consume all 100 training rows once. The worker bound is
300 seconds plus 5 seconds termination grace; no representative-data throughput
or memory qualification currently promises completion. Padded training tokens
plus three full padded validation traversals (initial, upstream pre-last-update,
saved-final reload) give an approximate work budget; divide by available worker
seconds after measured preparation overhead to assess throughput feasibility.

The original 128-token/three-update fixture does not qualify either selection.
Do not fabricate a captured-runtime/source profile or circular source-SHA
admission. Root independently reviews final pushed source, runtime capture,
complete rows/masks and finite controls; only root supplies original signed
single-use admission. Preserve all prior claims and failures, and never replay
training to recover preparation, readback or evaluation errors. Completion and
held-out loss are workflow evidence, not a writing-quality or reward evaluation.
