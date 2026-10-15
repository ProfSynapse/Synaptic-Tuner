# HF SFT qualification host fixture

This dedicated host branch is not an engine PR. Its engine Gitlink is exactly
038a62984a4745574d1eaf739a77272a62bb6deb, with the actual pushed
`release/stable-qualification-fixes` branch hint. Verify its actual public
branch HEAD before authenticating a new source proof. Do not substitute `main`
for this exact selected source.
The host itself must be reviewed/pushed and its exact branch verified before
source authentication; local creation is not pushed proof.

Eight complete authored Question/Answer documents live in data/hf-sft-v1/sources.
Actual public CLI ingestion and dataset preparation produced the retained
publication under data/hf-sft-v1/prepared. Every question and literal answer is
present, with six training and two validation rows. This raw-text/v1 lane uses
full-token loss over both; it does not relabel the separate v2 metrics fixture,
use a chat template or claim completion-only masking.

Reproduce locally from this host root using the installed accepted interpreter:

```sh
python synaptic-tuner/tuner.py ingest --config configuration/hf-sft-v1/ingestion.json --select fixture=data/hf-sft-v1/sources --json
python synaptic-tuner/tuner.py prepare-dataset --config configuration/hf-sft-v1/preparation.json --json
```

The exact bundle is cbad3e6be421470a8f3480944f95e52f151ec315d0914472fee27e2449265a1b;
prepared identity is 7aa26aba079ac1210466c5c2cdf747f00c764b5bc4ed366e2f67af7ffa0d3770;
2,821 original JSONL bytes hash to 739b3df17bf7fc39d5c9487136a633e8ad2703e700271e224756df8074d2c361.
The normalized config selects named qwen35-sft-v1/model-tokenizer revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a and three steps. Configuration is not
HF image/GPU/model/update qualification.

Portable preparation templates intentionally retain unresolved absolute paths,
source-key references, actual bucket, fresh prefixes, live quote and current
host-grant inputs. They are not operational consumer config or ready evidence.
Use the accepted host-owned prepare_hf.py example from PR242 alongside the
installed engine, using the same final engine source and installed accepted wheel.
Its genuine source verifier and compiler must accept actual completed inputs
before it publishes consumer.json. Do not copy fabricated quote hashes or bucket
IDs into the template. Named source key remains distinct from HF_TOKEN; original
authority/reader keys must remain stable for source-free known-run recovery.

Read-only pinned Hub1.33 CLI on 2026-10-14 confirmed professorsynapse, memberships
SynapticLabs and unsloth-jobs, no buckets listed under SynapticLabs, and existing
personal buckets professorsynapse/toolset-training-bootstrap and
professorsynapse/toolset-training-artifacts, plus org bucket
unsloth-jobs/jobs-artifacts. The owner selected the personal namespace professorsynapse and existing private
professorsynapse/toolset-training-artifacts bucket. This selection does not prove
empty attempt prefixes, available credits or current write/job permission.
A fresh finite quote remains part of operational admission. The repo
bucket CLI refused before provider access due to optional transformers import;
no dependencies/runtime workaround was added. Official pinned hf buckets list
was used instead, with named credentials through explicit dotenv CLI.

This host fixture does not itself execute jobs, upload buckets, download models
or authorize training. Earlier consumed attempts are retained separately and do
not qualify this source. Existing capability defaults and no-replay rules remain.

This candidate rebind preserves the original 7df0a0bf host history and exact
eight authored rows. The current engine includes shared prepared-SFT validation, finite ordinary-HF
failure exits and raw command-argument transport, plus the installed metrics
source-planning correction. Explicit bf16/fp16 keys are absent: this fixture uses
the trainer's existing adaptive precision behavior, rather than claiming an
explicit portable precision control. Raw packing remains false.
The selected package version remains 1.1.0; final qualification is still pending. Verify its
actual pushed branch and this host ref before source authentication. CI and live
qualification are separate; this fixture alone grants no effects. Owner selected private professorsynapse/toolset-training-artifacts, but no job/upload/model download is authorized by this host. Fresh local source/authority/reader keys and caller config remain private, outside tracked files.

The fresh 2026-10-15 replacement proposal is sequential: complete and natively
verify the three-step job before submitting the separate 32-step cancellation
job. Stop the remaining unused scope if completion fails or is indeterminate. Both descriptors opt into cancellation before plan for bounded queued cleanup: after a 15-minute queue cutoff, either queued job may receive one cancellation command, with at most one per known job total. A queued cleanup is not active-cancellation qualification. Publishing this host does not itself grant spending or execution authority. Earlier consumed attempts remain preserved.
