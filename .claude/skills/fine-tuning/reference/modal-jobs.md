# Modal Jobs Reference

Modal is a serverless-container cloud lane for GPU jobs: define an `@app.function`
that runs your work inside a pinned image on a Modal-provisioned GPU. Use it when
a job needs a GPU class or an on-demand elasticity that the RunPod wrapper lane
(`reference/runpod-jobs.md`) does not give you, or as a second independent
provider when HF Jobs and RunPod are both flaky.

Like the RunPod lane, this is for arbitrary wrapper work. Cloud TRAINING through
the tuner still goes via `tuner.py cloud-run` / `cloud-pipeline`; see
`reference/cloud-training.md`.

---

## Launch: always `--detach` for real jobs

A plain `modal run <app>` ties the app's lifetime to the launching client. When
that client goes away -- session end, dropped connection, killed babysitter --
Modal sends the app a cancellation signal and the run dies; the app is created as
an "ephemeral" app for exactly this reason.

For any job longer than a quick smoke, launch detached so the run outlives the
client:

```bash
modal run --detach <app_module>::<function>
```

Monitor a detached run by its app id:

```bash
modal app logs <app-id>
modal app list          # find the app id
```

### `--detach` alone does not survive a graceful client death

`--detach` protects the APP from client disconnect, but not the in-flight call.
If the client process receives a graceful signal (SIGINT/SIGTERM) while blocked
on `.remote()`, the unwinding call sends an explicit input-cancel RPC -- the log
shows "Received a cancellation signal while processing input" -- and the running
function dies even though the app was detached. Only SIGKILL or a network drop
leaves the input running.

The robust fix is to remove the blocking client entirely: have the local
entrypoint use `.spawn()` instead of `.remote()`.

```python
@app.local_entrypoint()
def main():
    call = my_fn.spawn(...)   # returns immediately after scheduling
    print(f"spawned {call.object_id}")
```

With `modal run --detach` + `.spawn()`, the client exits on its own within
seconds and there is never an in-flight input for a dying client to cancel.
Completion is then observed out-of-band: `modal app logs` plus a DONE marker on
the checkpoint Volume (see the crash-proof pattern below). Note Modal's caveat
that detach keeps only the LAST triggered function alive -- one spawn per
`modal run` invocation.

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
