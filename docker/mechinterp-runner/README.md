# MechInterp runner image

A pinned CUDA runtime for local GPU experiment work: activation
reading/writing, probe fitting, and small-model inference on a single
consumer GPU. This image is intentionally generic. It carries no project
paths, datasets, or configs; a calling project mounts its own workspace and
code at run time.

## Why this exists

Local GPU runs previously used a shared, hand-maintained conda environment.
That environment aged out silently: it kept working for older model
architectures while its `transformers` version fell behind what newer
architectures needed, and the mismatch was only caught mid-experiment. Files
that feed an experiment are already sha256-pinned in `experiment.yaml`, but
the runtime that executes them was not pinned at all, which left a gap in the
provenance story. This image closes that gap for the local lane the same way
the cloud lane is already closed: pin the runtime, print what it actually is
at start time, and record that alongside the file pins.

## What is pinned

| Component | Version |
| --- | --- |
| Base image | `nvidia/cuda@sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9` (`12.8.1-runtime-ubuntu22.04`) |
| Python | 3.10 (from the base image's Ubuntu 22.04 packages) |
| torch | `2.9.1` (PyTorch CUDA 12.8 wheel index) |
| transformers | `5.12.1` |
| flash-linear-attention | `0.5.1` |
| safetensors | `0.8.0` |
| scikit-learn | `1.7.2` (newest release supporting Python 3.10; 1.8+ needs >=3.11) |
| numpy | `2.5.1` |
| pyyaml | `6.0.3` |
| huggingface_hub | `1.23.0` |

Every pin is an exact version, not a floor or a range. When a new
architecture needs a newer runtime, bump the pins here in a reviewed change
rather than upgrading packages inside a running container.

## Build

```bash
cd docker/mechinterp-runner
docker build \
  --build-arg MECHINTERP_RUNNER_GIT_REVISION="$(git rev-parse HEAD)" \
  -t mechinterp-runner:local \
  .
```

## Capture the digest

A locally built image that has not been pushed to a registry has a content
digest (its Image ID) but no `RepoDigest`, since a `RepoDigest` is only
assigned once a registry has the content. Either is a valid provenance
reference; use whichever is available:

```bash
# Local build (no registry push): use the Image ID.
docker image inspect mechinterp-runner:local --format '{{.Id}}'

# After a registry push/pull: use the RepoDigest instead.
docker image inspect <registry>/mechinterp-runner:<tag> --format '{{index .RepoDigests 0}}'
```

## Run pattern (WSL2 + NVIDIA)

```bash
docker run --rm -it \
  --gpus all \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(pwd):/workspace" \
  --env IMAGE_DIGEST="$(docker image inspect mechinterp-runner:local --format '{{.Id}}')" \
  --env HF_TOKEN \
  mechinterp-runner:local \
  python your_script.py
```

Notes on that invocation:

- `--gpus all` requires the NVIDIA Container Toolkit registered with the
  Docker daemon (`docker info` should list an `nvidia` runtime). Confirm this
  before assuming a run has GPU access; a missing runtime fails at container
  start, not silently.
- The Hugging Face cache mount is read-write so repeated runs reuse
  downloaded weights instead of re-fetching them per container.
- `--env HF_TOKEN` (with no `=value`) passes the token through from the
  calling shell's environment without baking it into the image or into any
  committed file. Export it first, e.g.
  `export HF_TOKEN=$(sed -n 's/^HF_TOKEN=//p' .env | tr -d '"\r\n')`, and
  never `echo` it.
- Mount the workspace read-write only when the run needs to write outputs
  into it; a read-only mount (`-v "$(pwd):/workspace:ro"`) is safer when it
  does not.

## Provenance line

The entrypoint runs `print_provenance.py` before the container command and
prints one line of JSON to stdout, for example:

```json
{"event": "mechinterp_runner_provenance", "image_digest": "sha256:...", "image_git_revision": "86b134c...", "torch": "2.9.1+cu128", "cuda_available": true, "cuda_version": "12.8", "transformers": "5.12.1", "python": "3.10.12"}
```

`image_digest` reflects whatever `IMAGE_DIGEST` the caller passed at `docker
run` time (see above); without it, the line reports `"unknown"` rather than
guessing, since a digest cannot be recovered from inside a running container.
`image_git_revision` is baked in at build time from the `git rev-parse HEAD`
passed as `MECHINTERP_RUNNER_GIT_REVISION`, so the exact Dockerfile that
produced the image is always recoverable even when the digest was not
recorded.

## How downstream projects should record the digest

A downstream project (for example, an experiment's `experiment.yaml`) should
capture, alongside its existing file hashes:

- the image digest (Image ID or RepoDigest, per above),
- the `image_git_revision` from the provenance line, and
- the full provenance JSON line itself, appended to the run log.

That gives a run record two independent ways to reconstruct the exact
runtime: the digest for the content, and the git revision for the
human-readable Dockerfile history behind that content.
