# Trainers/ace_step/Dockerfile
#
# SEPARATE purpose-built image for the ACE-STEP v1.5 music / diffusion stack.
# Build contract docs/architecture/ace-step-pipeline-contract.md §9 + §Cross-cutting.
#
# WHY A SEPARATE IMAGE (not an overlay on unsloth/unsloth:latest):
#   The repo's text trainers run transformers 5.x; ACE-STEP v1.5 requires
#   transformers>=4.51,<4.58 (empty intersection — VERIFIED conflict). The audio
#   stack (cu128 torch 2.10, Lightning Fabric, peft/LyCORIS, diffusers, torchaudio
#   + ffmpeg) is a disjoint dependency world, so it lives in its own image and
#   never perturbs another method's transformers pin.
#
# PIN DISCIPLINE (repo convention — embedding TBD-PENDING-SMOKE precedent):
#   The exact wheel set is captured from the FIRST working image build + preprocess
#   smoke, NOT invented. ACE-STEP's own pyproject.toml (at the pinned ref below)
#   pins the cu128 wheels via its uv index, so installing ACE-STEP from its repo
#   is the reproducible source of truth — this Dockerfile defers to it.
#
# ACE-STEP REPO PROVISIONING (§9): cloned at a PINNED ref INTO the image (NOT
#   vendored into Synthetic Conversations). The build-host clone for the §1.3 /
#   §7 `--help` byte-confirm lives at ./vendor/ACE-Step-1.5 (gitignored) and is a
#   SEPARATE artifact from this in-image clone.
#
# ACE-STEP v1.5 source pinned to the CANONICAL COMMIT SHA (preparer-acestep #32,
#   confirmed from live GitHub). The SHA is IMMUTABLE — git tags can be moved, so
#   we clone the tag's branch then hard-reset to the SHA to guarantee a
#   reproducible build. SHA dce621408bee8c31b4fcf4811682eb9359e1bc94 == tag v0.1.8
#   (verified to match the build-host vendor/ACE-Step-1.5 clone).
#
# HF WEIGHTS (pulled at RUN time, not in this image): the default aggregate repo
#   is ACE-Step/Ace-Step1.5 (note the irregular casing) @ revision
#   19671f406d603126926c1b7e2adc169acbcade22 (license mit). It bundles VAE +
#   text-encoder + default turbo DiT + default 1.7B LM. ⚠️ Upstream
#   model_downloader.py pins NO HF revision (pulls `main`), so the wrapper /
#   model_registry MUST pass revision=<sha> to snapshot_download for a
#   reproducible pull — that pin lives in backend-coder-2's model_registry.yaml.

# CUDA 12.8 devel base (provides the cu128 toolkit ACE-STEP's wheels target).
#
# IMMUTABLE BASE PIN (review F-3): the base is pinned by @sha256 MANIFEST-LIST
# digest, not just the floating tag — nvidia re-publishes the 12.8.0 tag, so a
# bare tag is not reproducible. The digest is the multi-arch manifest list, so
# Docker still selects the correct per-arch image at pull. The tag is kept
# alongside the digest purely for human readability; when both are present Docker
# resolves by digest and the tag is ignored. Digest resolved 2026-06-26 from the
# Docker Hub registry (docker-content-digest of the manifest list). To re-pin on a
# deliberate base bump: `docker manifest inspect nvidia/cuda:<tag>` or read the
# registry docker-content-digest header, then update BOTH the tag and the digest.
#
# SCOPE NOTE: only the BASE image is pinned here. The cu128 PIP wheel set
# (transformers / torch / torchaudio / diffusers …) remains TBD-FROM-SMOKE — those
# exact pins genuinely require a CUDA image build to capture and stay deferred to
# the TEST/CI pin-from-smoke step (review F-3 is intentionally base-digest only).
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04@sha256:c900a2ac7f3d860c854f7364eadfe32b8cd33933a838f96b71e9bb6525f77a8c

# ACE-STEP v1.5 pinned ref. ACE_STEP_SHA is the immutable canonical commit;
# ACE_STEP_REF is the tag used to fetch the branch before resetting to the SHA.
ARG ACE_STEP_REF=v0.1.8
ARG ACE_STEP_SHA=dce621408bee8c31b4fcf4811682eb9359e1bc94
ARG ACE_STEP_REPO=https://github.com/ace-step/ACE-Step-1.5.git

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# --- System packages ---
# ffmpeg is a SYSTEM package here (NOT pip) — torchaudio needs the ffmpeg
# libraries to decode mp3/wav/flac/ogg audio. git is needed to clone ACE-STEP.
# python3.11 matches ACE-STEP's requires-python >=3.11,<3.13.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

# --- ACE-STEP v1.5 (cloned at the pinned ref, installed from its own pyproject) ---
# Installing ACE-STEP from its repo pulls its pyproject-pinned cu128 torch/torchaudio
# (via its [tool.uv] index) + transformers<4.58 + diffusers + lightning + peft +
# lycoris-lora + the DCAE/codec deps (vector-quantize-pytorch, torchcodec, torchao,
# pytorch-wavelets). This keeps the image in lockstep with the provisioned ref.
# The cu128 extra-index is supplied so pip resolves the +cu128 wheels ACE-STEP pins.
RUN git clone --branch ${ACE_STEP_REF} ${ACE_STEP_REPO} /opt/ace-step \
    && git -C /opt/ace-step reset --hard ${ACE_STEP_SHA} \
    && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        /opt/ace-step

# --- Our method-local island extras (only what ACE-STEP doesn't already pull) ---
# Most deps come transitively from ACE-STEP above; this layer adds the repo-standard
# config/HF utilities our wrapper + cloud dataset-pull lane need. Kept minimal so we
# don't shadow ACE-STEP's pinned versions.
COPY Trainers/ace_step/requirements.txt /tmp/ace_step_requirements.txt
RUN python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        huggingface-hub hf-transfer datasets pyyaml python-dotenv

# ACE-STEP's headless CLI (train.py preprocess/fixed, cli.py inference) lives at
# /opt/ace-step. The wrapper Trainers/ace_step/train_ace_step.py (mounted from the
# repo at /workspace/repo) shells out to it. PYTHONPATH lets `import acestep` resolve.
#
# ACE_STEP_HOME is the canonical seam (contract §9 container-path fix) that tells
# the wrapper WHERE the ACE-STEP CLI lives. In THIS image it is /opt/ace-step (the
# in-image clone). backend-coder-2's config_translation.py reads ACE_STEP_HOME and
# resolves train.py/cli.py under it; when ACE_STEP_HOME is unset (the build-host
# byte-confirm lane) it defaults to repo_root/vendor/ACE-Step-1.5. So the SAME
# wrapper code finds the CLI at /opt/ace-step inside the container and at vendor/
# on the host. Var name is fixed byte-for-byte by agreement with backend-coder-2.
ENV PYTHONPATH=/opt/ace-step:${PYTHONPATH} \
    ACE_STEP_HOME=/opt/ace-step
WORKDIR /workspace/repo

# Sanity default: show the ACE-STEP training entry's help. Overridden at run time
# by the recipe's run.command. (Built-time --help byte-confirm is contract §1.3.)
CMD ["python", "/opt/ace-step/train.py", "fixed", "--help"]
