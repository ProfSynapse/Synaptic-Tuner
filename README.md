# HF SFT qualification host fixture

This dedicated host branch is not an engine PR. Its engine Gitlink is exactly
4b77bd1f8e8a0abd664907a575166bfa41f11a7d, with the actual pushed
`release/grpo-model-publication-integration` branch hint. A credential-disabled
public ls-remote observed that exact branch HEAD during preparation. Do not use
`main` as its source hint: main has a different commit despite identical tree.
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
installed engine, since this exact runtime Gitlink predates that additive helper.
Its genuine source verifier and compiler must accept actual completed inputs
before it publishes consumer.json. Do not copy fabricated quote hashes or bucket
IDs into the template. Named source key remains distinct from HF_TOKEN; original
authority/reader keys must remain stable for source-free known-run recovery.

Read-only pinned Hub1.33 CLI on 2026-10-14 confirmed professorsynapse, memberships
SynapticLabs and unsloth-jobs, no buckets listed under SynapticLabs, and existing
personal buckets professorsynapse/toolset-training-bootstrap and
professorsynapse/toolset-training-artifacts. These are candidates, not selected
qualification destinations or verified empty attempt prefixes. Owner billing,
namespace/destination selection and fresh finite quote remain open. The repo
bucket CLI refused before provider access due to optional transformers import;
no dependencies/runtime workaround was added. Official pinned hf buckets list
was used instead, with named credentials through explicit dotenv CLI.

No Job, bucket creation/upload, model download, image build or paid training was
performed. Existing capability defaults and lost-reply/no-replay rules remain.
