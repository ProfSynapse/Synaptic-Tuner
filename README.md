# Synthetic GRPO qualification host

This dedicated host branch reuses the genuine archived smoke topology. It is not
an engine development branch and must not be merged into engine main. The engine
gitlink selects exact `af1cb98912cfbb446101526d41285ac39e54b034`.

`configuration/grpo.yaml` selects the checked-in unqualified SmolLM135 candidate,
three steps/four generations and deterministic blue regex reward. `data/rows.jsonl`
and its independently verified four-row `.synaptic/tracking/datasets` publication are synthetic fixtures;
`configuration/preparation.json` records exact original bytes/semantic identity.
All paths are host-relative. No provider IDs, old resources, credentials or
qualification receipts are retained here.

Use the installed pinned engine with `SYNAPTIC_ENGINE_ROOT` selecting the checked
out `synaptic-tuner` submodule. Existing `train --job-config configuration/grpo.yaml
--project-root . --plan --json` and `--prepare-local --json` are local preparation.
Keep actual run state under ignored `.synaptic/`, private directories0700/files0600.
A Git checkout does not itself establish private runtime permissions.

Public remote source verification must authenticate the actual pushed host branch
and exact engine gitlink before GPU training. Qualification and training retain
reviewed exact CI, current scope and bounded fresh-attempt requirements. Existing
explicit bounded session authorization remains applicable; the fixture and
configuration alone are not effect authority.

Fresh final source fixture; normal local plan/preparation only. No CPU image, GPU/model/update/reward qualification receipt is claimed. Earlier G and d494 work remain preserved.

This fresh local host selects the diagnostic candidate for review. Publishing this host does not itself grant execution authority. Earlier consumed runs and private state remain preserved.
