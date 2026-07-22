# Modal Jobs Reference

Modal is a serverless-container cloud lane for GPU jobs: define an `@app.function`
that runs your work inside a pinned image on a Modal-provisioned GPU. Use it when
a job needs a GPU class or an on-demand elasticity that the RunPod wrapper lane
(`reference/runpod-jobs.md`) does not give you, or as a second independent
provider when HF Jobs and RunPod are both flaky.

The generic SFT wrapper also supports exact external YAML plus private data on a
named input Volume. Inspect that path with
`python scripts/plan_modal_sft_job.py ...`; it hashes the local config and
dataset, validates YAML `dataset.local_file`, and prints staging and launch argv
without importing Modal or creating an app, Volume, or job.

Inspection requires a fully clean worktree whose local `HEAD` equals the
`origin/<branch>` tip (not merely an ancestor), an immutable OCI image digest,
and an exact pip overlay. Pip entries must be `name==version`, a URL bound by a
full `#sha256` digest, or a `git+https` URL pinned to a full commit. Credential-
bearing repository or package URLs are rejected before they can enter argv or
logs.

---

## Launch: always `--detach` for real jobs

A plain `modal run <app>` ties the app's lifetime to the launching client. When
that client goes away -- session end, dropped connection, killed babysitter --
Modal sends the app a cancellation signal and the run dies; the app is created as
an "ephemeral" app for exactly this reason.

For any job longer than a quick smoke, launch detached so the run outlives the
client:

```bash
modal run --detach <app_module>::<remote_function>
```

Monitor a detached run by its app id:

```bash
modal app logs <app-id>
modal app list          # find the app id
```

### Call the remote function directly

Do not route a governed run through a local entrypoint that calls `.remote()` or
`.spawn()`. That nested submission shape can either remain client-bound or leave
an app with zero running tasks. Target the decorated remote function directly:

```bash
modal run --detach Trainers/cloud/train_modal.py::run_training \
  --trainer-type sft \
  --repo-url <url> --repo-branch <branch> --repo-commit <full-sha>
```

After submission, require a non-empty app id and prove `run_training` has a
running or completed task with `modal app list --json` and
`modal app logs <app-id>`. An app record by itself is not launch evidence.

For private config-driven SFT, stage the inspected YAML and JSONL into the named
input Volume paths printed by `plan_modal_sft_job.py`. Pass the printed SHA-256
values to `run_training`; the remote wrapper re-hashes both files before model
load and fails closed if either byte stream or mounted path differs.
The wrapper writes `modal_job_provenance.json` before clone, upgrades it after
input verification, and transports the non-secret record into both
`manifest.json` and `training_lineage.json`. The record binds source commit,
input hashes/paths and Volume, runtime image/pip/GPU/timeout, artifact Volume,
and cache Volume. On a nonzero trainer exit—including adapter or merged-model
qualification failure—the wrapper changes the canonical manifest to failed and
commits the output Volume before touching the replaceable model cache.

The interactive `python tuner.py cloud` route intentionally remains blocking:
it calls the remote function directly without `--detach`, waits for its terminal
exit, and only then lets the handler report completion. Detached submission is
reserved for the inspected command emitted by `plan_modal_sft_job.py`, where the
operator separately verifies a real task.

---

## Image setup gotchas

- **Clear a hijacking ENTRYPOINT.** Images that ship a process supervisor (the
  Unsloth images run `supervisord`) will hijack the container and never run your
  function. Reset the entrypoint when building the image:
  `image = base_image.entrypoint([])`.
- **Bake the hf_xet mitigation into the image env.** The `hf_xet` CAS backend
  hangs without a timeout on multi-GB model pulls (see gotcha #5 in
  `reference/runpod-jobs.md`); it is a `huggingface_hub` issue, not provider
  specific, so Modal hits it too. Set both in the image env (or the function's
  secrets):

  ```python
  image = image.env({
      "HF_HUB_DISABLE_XET": "1",
      "HF_HUB_ENABLE_HF_TRANSFER": "0",
  })
  ```

---

## Crash-proof long-run pattern

Modal containers can die mid-run (preemption, OOM, node loss). A long job must be
able to survive a container death and resume rather than restart from zero. The
pattern that holds up:

1. **Write outputs to container-local disk first.** Fast, simple; the local disk
   is scratch and disappears with the container.
2. **Mirror to a `modal.Volume` on a background daemon thread.** A ~120s loop
   copies new/changed output files into the mounted Volume and calls
   `vol.commit()`. Catch and log every exception inside the loop -- a failed
   mirror tick must NEVER kill the run.
3. **Restore from the Volume before starting work.** At function entry, copy any
   prior outputs back from the Volume onto local disk so the native script's own
   resume logic engages (finds its last checkpoint / done markers and continues).
4. **Let container death respawn and resume.** Decorate the function with retries
   so a killed container comes back and re-enters step 3:

   ```python
   @app.function(retries=modal.Retries(max_retries=3, backoff_coefficient=1.0))
   ```

5. **Write a DONE marker at the very end.** A sentinel file (mirrored to the
   Volume) lets a respawn distinguish "finished, nothing to do" from "died
   partway, resume".

The Volume is checkpoint/scratch space, not the system of record: the final
result still has to land on the durable store (an HF staging repo) per the
artifact contract in `reference/runpod-jobs.md`. A long run that produces a real
model must opt into publishing it to the Hub -- the code default is off so a bare
`modal run` cannot silently publish.

---

## When to prefer Modal vs RunPod

- **RunPod wrapper lane** (`reference/runpod-jobs.md`): byte-pinned image + git
  commit on a specific single-GPU class; the launcher owns pod teardown. Good for
  reproducing a job that must match a known GPU exactly.
- **Modal**: serverless elasticity, native retries/Volumes for crash recovery,
  detached runs that survive the client. Good for long or bursty work, or a
  different GPU class.

Both lanes obey the same staging prerequisite and artifact contract (see
`reference/runpod-jobs.md`): referenced repos exist on the Hub at pinned
revisions before launch, and results land on the Hub as the durable record.
