# Local Mac Bucket To GGUF

Use this workflow when the user is on a Mac and wants to take a finished cloud-trained adapter from HF Bucket storage, merge it locally, quantize it to GGUF with `llama.cpp`, and optionally drop it into local software such as LM Studio or Ollama.

This is the right path when:
- the source of truth is a bucket-backed `final_model/` from HF Jobs
- the user wants a local `.gguf`, not another HF upload
- the machine already has `llama.cpp` built, or can build it locally

This is not the right path when:
- the user only wants to upload to Hugging Face
- the machine does not have enough free disk for a local merge
- the model is multimodal and you have not confirmed whether the adapter is language-only enough for local serving

## Workflow

1. Resolve the exact source adapter path.
2. Pull the bucket-backed `final_model/` locally.
3. Ensure a local merge environment exists.
4. Merge the adapter into a full local model.
5. Convert the merged model to a base GGUF.
6. Quantize only the format the user actually needs first.
7. Optionally copy the `.gguf` into LM Studio or create an Ollama `Modelfile`.

## Source Resolution

For cloud-trained runs, prefer the bucket-backed `final_model/` prefix, not the remote container filesystem.

Common ways to confirm the source:

```bash
python3 tuner.py bucket list --path runs/hf_jobs/sft/<run-prefix>/final_model/ --limit 20
```

For locally tracked experiments, `experiment.json` is often enough to recover the training artifact root:

```bash
python3 - <<'PY'
from pathlib import Path
import json
p = Path(".tracking/experiments/<experiment-id>/experiment.json")
data = json.loads(p.read_text())
print(data["artifact_roots"]["training"])
PY
```

## Pull The Adapter

Pull into a dedicated temp or staging directory, one model at a time:

```bash
python3 tuner.py bucket pull \
  --path runs/hf_jobs/sft/<run-prefix>/final_model \
  --dest /tmp/model-merge-staging/<slug>
```

The resulting local adapter usually lands at:

```text
/tmp/model-merge-staging/<slug>/toolset-training-artifacts/runs/hf_jobs/sft/<run-prefix>/final_model
```

Verify the adapter contains:
- `adapter_config.json`
- `adapter_model.safetensors`
- tokenizer files if available

## Disk Budgeting

Do not start blindly on a space-constrained Mac.

Check free space first:

```bash
df -h .
```

Practical guidance learned from this workflow:
- a 4B-class merged model can take roughly `7-8 GiB`
- the base GGUF can take roughly another `7-8 GiB`
- a `Q4_K_M` quant may still take `2-3 GiB`
- the downloaded adapter and cached base model add more overhead

If space is tight:
- do one model at a time
- create only `Q4_K_M` first
- delete temp adapter pulls and HF cache after a successful merge
- delete the merged model and base GGUF once the final quant is safely copied where the user wants it

## Merge Environment On macOS

The repo's canonical merge utilities use Unsloth, but on a Mac you may not have a working local Unsloth environment. A plain `transformers` + `peft` merge venv is an acceptable fallback for text models.

Example:

```bash
python3 -m venv --system-site-packages /tmp/model-merge-venv
source /tmp/model-merge-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel peft accelerate 'transformers>=4.58.0'
```

Why `--system-site-packages`:
- it reuses the existing local `torch`
- it minimizes reinstall cost on the Mac

Verify the env:

```bash
source /tmp/model-merge-venv/bin/activate
python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("mps", torch.backends.mps.is_available())
PY
```

Important:
- if `mps` is `False`, merges will be CPU-only
- CPU-only merges can still work for 4B-class models, but expect them to be slower
- if the model family is very new, the merge env and `llama.cpp` may need updating independently; a successful merge does not imply your local GGUF converter supports the architecture yet

## Merging

For local Mac fallback merges, use plain Transformers + PEFT against the canonical base model.

Do not blindly merge against a training-time alias like `unsloth/<model>-bnb-4bit` if the local workflow is plain Transformers. Prefer the canonical base model repo when known.

Example merge:

```bash
source /tmp/model-merge-venv/bin/activate
python - <<'PY'
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_path = Path("/tmp/model-merge-staging/<slug>/.../final_model")
output_path = Path("/tmp/model-merge-staging/<slug>/merged/<model-name>")
base_model = "Qwen/Qwen3-4B"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(model, str(adapter_path))
model = model.merge_and_unload()
output_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(output_path), safe_serialization=True)
tokenizer.save_pretrained(str(output_path))
print(output_path)
PY
```

Expect the merged output directory to contain at least:
- `config.json`
- `model.safetensors`
- tokenizer files

### Qwen 3.5 Caveat

Qwen 3.5 is an important special case on this repo owner's Mac workflow.

The adapter may be saved against:
- `Qwen3_5ForConditionalGeneration`

while a naive local load via `AutoModelForCausalLM` may resolve to:
- `Qwen3_5ForCausalLM`

Do not assume those are interchangeable for LoRA merge.

Before merging, inspect:
- `adapter_config.json`
- a few adapter tensor names from `adapter_model.safetensors`

If the tensor names live under `language_model.*`, use the conditional-generation wrapper for the merge path:

```bash
source /tmp/model-merge-venv/bin/activate
python - <<'PY'
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

adapter_path = Path("/tmp/model-merge-staging/<slug>/.../final_model")
output_path = Path("/tmp/model-merge-staging/<slug>/merged/<model-name>")
base_model = "Qwen/Qwen3.5-4B"

model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(model, str(adapter_path))
model = model.merge_and_unload()
output_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(output_path), safe_serialization=True)
tokenizer.save_pretrained(str(output_path))
print(output_path)
PY
```

Why this matters:
- if you merge Qwen 3.5 through the wrong wrapper, PEFT may emit large missing-key warnings and silently produce a suspect output
- using the matching conditional-generation wrapper gave a clean attach/merge for the adapter in this workflow

## llama.cpp Conversion

On this repo, the reliable converter lives in:
- `shared/upload/converters/gguf_reliable.py`

And a local `llama.cpp` build may already exist at:
- `Trainers/llama.cpp`

If the model is already merged, do not call the top-level `ReliableGGUFConverter.convert()` path. That flow assumes it starts from a LoRA adapter and will try to merge again.

For already merged models, use the lower-level methods directly:

```bash
source /tmp/model-merge-venv/bin/activate
python - <<'PY'
from pathlib import Path
from shared.upload.converters.gguf_reliable import ReliableGGUFConverter

merged_model = Path("/tmp/model-merge-staging/<slug>/merged/<model-name>")
out_dir = Path("/tmp/model-merge-staging/<slug>/converted/<model-name>/gguf")
out_dir.mkdir(parents=True, exist_ok=True)

base_gguf = out_dir / "<model-name>.gguf"
q4_gguf = out_dir / "<model-name>-Q4_K_M.gguf"

converter = ReliableGGUFConverter(
    llama_cpp_dir=Path("/Users/jrosenbaum/Documents/Code/Synthetic Conversations/Trainers/llama.cpp")
)

if not converter.convert_to_gguf_base(merged_model, base_gguf, dtype="bf16"):
    raise SystemExit(1)
if not converter.quantize_gguf(base_gguf, q4_gguf, "Q4_K_M"):
    raise SystemExit(2)

print(base_gguf)
print(q4_gguf)
PY
```

If conversion fails with:

```text
Model <architecture> is not supported
```

update the local `llama.cpp` checkout first:

```bash
git -C Trainers/llama.cpp pull --ff-only origin master
cmake --build Trainers/llama.cpp/build --config Release -j4
```

That was necessary for local Qwen 3.5 GGUF conversion on this Mac. The merge path worked before the converter path did.

Start with only:
- `Q4_K_M`

Then add `Q5_K_M` or `Q8_0` only if the user explicitly wants them and disk headroom exists.

## LM Studio

On this Mac, LM Studio stores local models under:

```text
~/.lmstudio/models
```

The observed working layout is:

```text
~/.lmstudio/models/<publisher>/<model-folder>/
  <model-name>-Q4_K_M.gguf
  config.json              # optional but helpful
```

Example:

```bash
mkdir -p "$HOME/.lmstudio/models/professorsynapse/qwen3-4b-toolcall-best"
cp "/tmp/.../Qwen3-4B-toolcall-best-Q4_K_M.gguf" \
   "$HOME/.lmstudio/models/professorsynapse/qwen3-4b-toolcall-best/"
cp "/tmp/.../config.json" \
   "$HOME/.lmstudio/models/professorsynapse/qwen3-4b-toolcall-best/"
```

Then:
- refresh My Models in LM Studio
- or restart LM Studio if it does not immediately pick up the new folder

## Ollama

For Ollama, create a minimal `Modelfile`:

```text
FROM /absolute/path/to/<model-name>-Q4_K_M.gguf
```

Then:

```bash
ollama create <tag-name> -f Modelfile
ollama run <tag-name>
```

## Cleanup

After a successful local install:
- keep the final quantized `.gguf`
- optionally keep `config.json`
- remove:
  - the staged adapter pull
  - the HF cache for the base model
  - the merged model directory
  - the base GGUF if the user only needs the quantized file

Typical cleanup candidates:

```bash
rm -rf /tmp/model-merge-staging/<slug>/.../final_model
rm -rf ~/.cache/huggingface/hub/models--<org>--<model>
rm -rf /tmp/model-merge-staging/<slug>/merged/<model-name>
rm -f  /tmp/model-merge-staging/<slug>/converted/<model-name>/gguf/<model-name>.gguf
```

Only remove files after the final quantized output has been copied into the user's target location.

## Learned Constraints

- One-model-at-a-time is the correct default on a Mac with limited free space.
- The repo's top-level reliable GGUF converter is best when starting from a LoRA adapter in an environment with Unsloth.
- For local Mac fallback workflows, it is often simpler to:
  - pull adapter
  - merge with plain Transformers + PEFT
  - convert merged weights with `llama.cpp`
- The HF bucket pull may reuse stale empty local prefixes. If a pull looks wrong, re-pull into a fresh destination.
- If the user wants to test immediately, prioritize one good `Q4_K_M` output over a full quantization bundle.
