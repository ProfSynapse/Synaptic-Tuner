# Synaptic Tuner

Synthetic dataset generation and LLM fine-tuning engine. Teacher models generate
training data, which is used for SFT/KTO fine-tuning of smaller models. The
engine is consumed as a git submodule by host projects; hosts own configuration,
credentials, and mutable state.

## How to work here

- `AGENTS.md` carries the standing engineering discipline (fine-tuning workflow,
  config-first generation, HF Jobs, Modal). Read it before changing those areas.
- `.skills/` is the canonical skill source; `.agents/skills` and `.claude/skills`
  are generated mirrors. After editing a skill run
  `python3 .skills/scripts/sync_skill_trees.py`, then `--check`.
- Load the relevant skill before starting: `fine-tuning` for training, cloud
  jobs, evaluation runs, experiments and dataset publishing;
  `synethetic-data-generation` for synthetic data; `evaluation` for the eval
  system; `upload-deployment` for model upload; `research-reporting` for
  research notes.
- Prefer existing CLIs, checked-in scripts and skills over ad hoc scripts or
  manual API probing. If a capability is missing, add it as a checked-in
  workflow and update the skill rather than leaving a one-off.
- Docs entry points: `docs/common-tasks.md` (how to run things),
  `docs/troubleshooting.md` (errors and recovery), `docs/project-reference.md`
  (scripts, configs, env vars, data formats).
- Discover resources with the CLI rather than from memory: `./run.sh status`,
  `./run.sh doctor`, `./run.sh list <resource>` (`python tuner.py` directly, or
  `.\run.ps1` on Windows).

## Rules

- Never write outputs to `/tmp`. Generated files stay in the repository
  (`docs/`, `Datasets/`, `scratch/`); test outputs go under `scratch/`.
- Stop early on errors. Watch output and kill a run as soon as it looks wrong.
- No hardcoding of scenario-specific behavior in SynthChat. Tool-call formats,
  workspace structures and label mappings come from config under
  `SynthChat/config/`. The bundled `useTools` wrapper is a toy example, not the
  canonical format.
- No backward-compatibility shims. Move code and update imports directly; no
  re-exports, dual signatures or deprecated wrappers.
- Secrets are referenced by name only and resolved at execution time. Never put
  credential values in code, configs, logs, argv or commit messages.
- Do not record session state, progress or status in this file. It holds only
  durable working rules; everything time-bound belongs in `docs/`.
