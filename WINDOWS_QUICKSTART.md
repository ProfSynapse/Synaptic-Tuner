# Windows Unsloth - Quick Start

## Setup (One Time)

```powershell
# 1. Create environment
conda create -n unsloth_env python=3.11 -y
conda activate unsloth_env

# 2. Run setup script
.\setup_unsloth_windows.ps1
```

**Time:** 10-15 minutes

---

## Training (Every Time)

### Quick Test Run

```powershell
conda activate unsloth_env
python train_windows.py --config small --dry-run
```

### Full Training

```powershell
conda activate unsloth_env
python train_windows.py --config 7b --dataset syngen_tools_11.14.25.jsonl
```

### Custom Configuration

```powershell
python train_windows.py --config 7b --dataset my_data.jsonl --output ./my_model --max-steps 500
```

---

## Configuration Options

**Pre-configured sizes:**
- `small` - Fast test (50 steps, 512 seq length)
- `7b` - Full 7B model (2048 seq length)
- `13b` - 13B model (1024 seq length, needs more VRAM)

**CLI options:**
- `--dataset PATH` - Your JSONL dataset file
- `--output DIR` - Output directory (default: ./unsloth_output)
- `--max-steps N` - Maximum training steps
- `--dry-run` - Just load model, don't train

---

## Dataset Format

Your JSONL file should have this format:

```json
{
  "messages": [
    {"role": "user", "content": "Your question here"},
    {"role": "assistant", "content": "Expected response"}
  ]
}
```

---

## Files You Need

**Required:**
- ✅ `unsloth_windows_patch.py` - Compatibility patches
- ✅ `windows_training_config.py` - Training configuration
- ✅ `train_windows.py` - Main training script
- ✅ Your dataset JSONL file

**Optional:**
- `test_unsloth_working.py` - Verify installation
- `.env` - For HuggingFace token

---

## Customizing Training

Edit `windows_training_config.py` to change:
- Learning rate
- Batch size
- LoRA parameters
- Sequence length
- And more...

Or create your own config:

```python
from windows_training_config import WindowsTrainingConfig

config = WindowsTrainingConfig(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    learning_rate=1e-4,
    max_seq_length=4096,
    # ... your settings
)
```

---

## Troubleshooting

**"CUDA not available":**
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

**"Module not found":**
```powershell
conda activate unsloth_env
pip install tensorboard  # For training logs
```

**Patches not applied:**
Make sure `unsloth_windows_patch.py` is in the same directory as `train_windows.py`

---

## Next Steps

After training completes:

1. **Test your model:**
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained("./unsloth_output")
   ```

2. **Upload to HuggingFace:**
   ```powershell
   huggingface-cli upload ./unsloth_output
   ```

3. **Use for inference:**
   ```python
   # Your inference code here
   ```

---

**Need help?** Check `UNSLOTH_WINDOWS_GUIDE.md` for detailed documentation.
