FROM vllm/vllm-openai:gemma4

ARG TRANSFORMERS_VERSION=5.5.0

USER root

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

RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps "transformers==${TRANSFORMERS_VERSION}" \
    && python3 -m pip install --no-cache-dir --upgrade pyyaml wandb hf_transfer python-dotenv rich peft

# Keep the runtime on the PyTorch path only. The Gemma vLLM images currently
# behave better when TensorFlow backend probing is disabled explicitly.
ENV USE_TORCH=1
ENV USE_TF=0

ENV HF_HUB_ENABLE_HF_TRANSFER=1
