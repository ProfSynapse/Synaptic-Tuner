FROM vllm/vllm-openai:gemma4

ARG TRANSFORMERS_SOURCE=https://github.com/huggingface/transformers/archive/refs/heads/main.zip

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# The stock gemma4 image currently ships a broken Transformers install that
# still exposes removed tensorflow_text backend references. Remove it fully
# before reinstalling a clean source build.
RUN python3 -m pip uninstall -y transformers || true
RUN python3 - <<'PY'
import pathlib
import shutil
import site

for base in site.getsitepackages():
    site_root = pathlib.Path(base)
    for pattern in ("transformers", "transformers-*.dist-info", "transformers-*.egg-info"):
        for path in site_root.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
PY

RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps "${TRANSFORMERS_SOURCE}" \
    && python3 -m pip install --no-cache-dir --upgrade pyyaml wandb hf_transfer python-dotenv rich

ENV HF_HUB_ENABLE_HF_TRANSFER=1
