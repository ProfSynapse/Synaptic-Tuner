# Generic CUDA runner for MechInterp-style pipelines: activation
# reading/writing, probe fitting, and small-model inference/generation on a
# single consumer GPU (validated against an RTX 3090 under WSL2). Intentionally
# project-agnostic: no experiment paths, datasets, or configs baked in. Any
# project that needs this runtime mounts its own workspace and code at run
# time (see README.md).
#
# Base pinned by digest so the OS/CUDA userspace layer is reproducible
# independent of what "ubuntu22.04" resolves to later.
FROM nvidia/cuda@sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9

ARG MECHINTERP_RUNNER_GIT_REVISION=unknown
ENV MECHINTERP_RUNNER_GIT_REVISION=${MECHINTERP_RUNNER_GIT_REVISION}

# transformers is parameterized because an instrument validated against one
# modeling implementation is not automatically valid under another: internal
# model attributes (e.g. Gemma4's KV-sharing layout) change between releases,
# and a consumer whose patches/hooks read those internals must run under the
# version it was validated against. The default stays the image's standard
# stack; a consumer overrides it at build time and records the resulting
# image digest plus this version in its run provenance.
ARG TRANSFORMERS_VERSION=5.12.1
ENV MECHINTERP_RUNNER_TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv

# Every package below is pinned to an exact version. torch comes from the
# PyTorch CUDA 12.8 wheel index to match the base image's CUDA runtime;
# everything else resolves from PyPI. Versions were chosen to match the
# already-validated Qwen3.5/Gemma-4 local stack (transformers 5.12.1, see
# docs/sessions/0030 in the research repo) rather than picking latest-on-date.
# scikit-learn is capped at 1.7.2, the newest release that still supports
# Python 3.10 (1.8+ requires >=3.11); numpy is capped at 2.2.6 for the same
# reason (2.3+ requires >=3.11, 2.5+ requires >=3.12); the base image ships
# Ubuntu 22.04's stock Python 3.10. accelerate is required by transformers'
# device_map= loading path (from_pretrained raises without it); 1.14.0 is the
# newest release this venv's Python 3.10 pip resolves.
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip==26.1.2 \
    && /opt/venv/bin/pip install --no-cache-dir \
        torch==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/venv/bin/pip install --no-cache-dir \
        transformers==${TRANSFORMERS_VERSION} \
        flash-linear-attention==0.5.1 \
        safetensors==0.8.0 \
        scikit-learn==1.7.2 \
        numpy==2.2.6 \
        pyyaml==6.0.3 \
        huggingface_hub==1.23.0 \
        accelerate==1.14.0

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY print_provenance.py /usr/local/bin/print_provenance.py
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
