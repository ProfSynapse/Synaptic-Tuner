# Troubleshooting Guide

Diagnostics, common issues, and recovery procedures.

---

## Quick Health Check

```bash
./run.sh doctor          # Full system diagnostics
./run.sh doctor --fix    # Auto-fix common issues
```

## Quick Reference

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| "CUDA not available" | PyTorch/CUDA mismatch | Reinstall PyTorch |
| "Connection refused" | Backend not running | Start LM Studio/Ollama |
| "Module not found" | Wrong environment | `conda activate toolset` |
| "Permission denied" | File permissions | `chmod +x script.sh` |
| "Out of memory" | Batch too large | Reduce `--batch-size` |
| "Loss is NaN" | Learning rate too high | Reduce LR by 10x |
| "No examples found" | Wrong file format | Check JSONL format |
| "API key invalid" | Token expired/wrong | Update `.env` file |

---

## Common Issues and Fixes

### CUDA / GPU Issues

**"CUDA not available"**
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Fix: Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**"CUDA out of memory" (OOM)**
```bash
# Option 1: Reduce batch size
python train_sft.py --model-size 7b --batch-size 4

# Option 2: Use smaller model
python train_sft.py --model-size 3b

# Option 3: Enable gradient checkpointing (in config)
# Option 4: Clear GPU memory
nvidia-smi --gpu-reset
```

### LLM Backend Issues

**"LM Studio not reachable"**
```bash
# Check if LM Studio is running
curl http://localhost:1234/v1/models

# WSL users: Use Windows host IP instead of localhost
# Find Windows IP: cat /etc/resolv.conf | grep nameserver
# Update .env: LMSTUDIO_HOST=<windows_ip>

# Ensure "Serve on Local Network" is enabled in LM Studio settings
```

**"Ollama connection refused"**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Check available models
ollama list
```

**"OpenRouter API error"**
```bash
# Verify API key is set
echo $OPENROUTER_API_KEY

# Test API connectivity
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

### Dataset Issues

**"Dataset validation failed"**
```bash
# Run validation to see specific errors
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py <file>

# Common fixes:
# - Check JSON syntax (missing commas, quotes)
# - Ensure "conversations" array exists
# - Verify role is "user" or "assistant"
# - Check "label" field for KTO datasets
```

**"No examples found in dataset"**
```bash
# Check file is not empty
wc -l <dataset_file>

# Check file format (should be JSONL)
head -1 <dataset_file> | python -m json.tool
```

### Training Issues

**"Training logs not appearing"**
- Check `logs/training_latest.jsonl` exists in run directory
- Verify `run_dir` path in callbacks configuration
- Check disk space: `df -h`

**"Loss is NaN"**
- Learning rate too high - reduce by 10x
- Data contains invalid values - run validation
- Gradient explosion - enable gradient clipping

**"Training stuck / no progress"**
- Check GPU utilization: `watch nvidia-smi`
- Verify data loader is working
- Check for deadlocks in multi-GPU setup

### Environment Issues

**"Missing dependencies"**
```bash
./setup_env.sh              # Full environment setup
./run.sh doctor --fix       # Or with auto-fix
pip install -r requirements.txt  # Manual
```

**"Module not found"**
```bash
conda activate toolset
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Local Docker Run Issues

**"Artifacts owned by root after local-run"**
- Symptom: `rm -rf toolset-training-artifacts/runs/local_docker/...` fails with `EACCES` / "Operation not permitted".
- Cause: `job.user: root` was set, or the chown-back trap failed mid-run.
- Fix: set `job.user: auto` in the job YAML (default), or reclaim ownership manually: `sudo chown -R $USER:$USER <path>`.

**"chown ineffective on WSL drvfs"**
- Symptom: runner prints a notice that the repo is on WSL drvfs (`/mnt/...`); artifacts still appear root-owned in Windows Explorer after exit.
- Fix: enable POSIX metadata on drvfs — add the following to `/etc/wsl.conf`, then run `wsl --shutdown` and reopen WSL:
  ```
  [automount]
  options="metadata"
  ```
  From WSL, chown-back now takes effect; Windows Explorer still shows its own overlay, which is expected.

**"Persistent container is holding VRAM"**
- Symptom: `nvidia-smi` shows GPU memory pinned by a `local-run-<name>` container after training finished.
- Cause: `job.persist: true` keeps the container alive between invocations so pip/HF-cache/triton-compile stay warm. The sleep-infinity idle process holds a small amount of VRAM when the image is CUDA-enabled.
- Fix: stop it when you're done iterating.
  ```bash
  python tuner.py local-run --job-config Trainers/recipes/<recipe>.yaml --stop            # stop but keep
  python tuner.py local-run --job-config Trainers/recipes/<recipe>.yaml --rm-persistent   # stop and delete
  ```
- Check state any time: `--container-status` prints `running` / `exited` / `absent`.

**"Zombie container after ctrl-C"**
- Symptom: after ctrl-C, the container still appears in `docker ps` and cannot be re-used; `docker exec` into it hangs.
- Cause: rare now. The runner uses `--init` (tini) as PID 1 inside persistent containers so SIGINT is forwarded cleanly to the training process and orphan children are reaped. A zombie implies tini itself stalled (docker engine bug) or you ran a container created before `--init` was added.
- Fix: `--rm-persistent` on the job-config to delete the container. If that hangs, restart Docker Desktop (Windows/WSL) or `sudo systemctl restart docker` (Linux), then re-run.

---

## Recovery Procedures

### Training Crashed Mid-Run

```bash
# 1. Find the last checkpoint
ls -la Trainers/sft/sft_output/<run_id>/checkpoints/

# 2. Check checkpoint integrity
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('<checkpoint_path>')"

# 3. Resume from checkpoint (if trainer supports it)
python train_sft.py --resume-from-checkpoint <checkpoint_path>

# 4. Or restart fresh with same config
python train_sft.py --config <original_config>
```

### Out of GPU Memory During Training

```bash
# Immediate fix: Kill process and clear memory
pkill -f train_sft
nvidia-smi --gpu-reset  # If needed

# Config fixes (in order of impact):
# 1. Reduce batch size (most effective)
# 2. Use gradient accumulation instead of large batch
# 3. Enable gradient checkpointing
# 4. Use smaller model variant (7B -> 3B)
# 5. Use 4-bit quantization for base model
```

### Evaluation Giving Weird Results

```bash
# 1. Verify model is fully loaded
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('<model_path>'); print(m)"

# 2. Check prompt format matches training format
# Compare: Evaluator/prompts/*.json with training data format

# 3. Verify sampling settings
# - Temperature: 0.0-0.3 for deterministic, 0.7-1.0 for creative
# - top_p: 0.9 typical
# - max_tokens: ensure sufficient for response

# 4. Test with a simple known-good prompt
python -c "
from transformers import pipeline
pipe = pipeline('text-generation', model='<model_path>')
print(pipe('Hello, how are you?', max_new_tokens=50))
"
```

### Dataset Improvement Not Working

```bash
# 1. Verify LLM backend is responding
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"test"}],"max_tokens":10}'

# 2. Check rubric exists
python -m SynthChat.services.rubric_runner --list

# 3. Validate input file format
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py <input_file>

# 4. Run with verbose logging
python -m SynthChat.services.rubric_runner \
  --file <input> --output <output> \
  --rubrics <rubric> --start-line 1 --end-line 1 \
  --verbose
```

### Upload to HuggingFace Failed

```bash
# 1. Verify HF_TOKEN is set and valid
python -c "from huggingface_hub import HfApi; HfApi().whoami()"

# 2. Check disk space for merge operation
df -h

# 3. Verify model path exists and is complete
ls -la <model_path>/

# 4. Try upload with smaller chunks
python3 .skills/upload-deployment/scripts/upload_model.py <model_path> <repo_name> \
  --save-method lora  # Upload just LoRA adapters first
```

### Synthetic Data Generation Errors

```bash
# 1. Check config is valid YAML
python -c "import yaml; yaml.safe_load(open('synth_chat/config/config.yaml'))"

# 2. Verify teacher model is accessible
# (depends on backend - LM Studio, OpenRouter, etc.)

# 3. Run with minimal config for testing
./Tools/run_synth_chat.sh --quick --dry-run

# 4. Check output directory is writable
touch Datasets/test_write && rm Datasets/test_write
```
