# MLX Fine-Tuning System for Mistral-7B-Instruct-v0.3

A production-ready, modular fine-tuning system for Mistral-7B-Instruct-v0.3 on Apple Silicon (Mac M4) using MLX framework with LoRA (Low-Rank Adaptation).

## Features

- **Optimized for Apple Silicon**: Uses MLX framework for efficient Metal GPU acceleration
- **LoRA Fine-Tuning**: Memory-efficient training with < 1% trainable parameters
- **Modular Architecture**: Clean separation of concerns with 6 core modules
- **Comprehensive Logging**: Structured logging with JSON output and metrics tracking
- **Checkpoint Management**: Automatic saving, resumption, and best model tracking
- **Memory Monitoring**: Real-time RAM and Metal GPU memory tracking
- **Data Validation**: Robust JSONL parsing with schema validation
- **Production-Ready**: Error handling, recovery mechanisms, and extensive documentation

## System Requirements

- **Hardware**: Mac with M1/M2/M3/M4 chip (16GB+ RAM recommended)
- **OS**: macOS 13.0+ (for Metal GPU support)
- **Python**: 3.9 or higher
- **Disk Space**: ~20GB for model + checkpoints

## Installation

### 1. Clone Repository

```bash
cd "/Users/jrosenbaum/Documents/Code/Synthetic Conversations/code"
cd mistral_lora_mac
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as package:

```bash
pip install -e .
```

### 4. Verify MLX Installation

```bash
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

Should output `True` if Metal GPU is available.

## Quick Start

### 1. Prepare Your Dataset

Place your JSONL dataset in the project directory or specify path in config:

```bash
# Example: Link dataset from parent directory
ln -s "../syngen_toolset_v1.0.0_claude.jsonl" syngen_toolset_v1.0.0_claude.jsonl
```

Dataset format (each line is JSON):
```json
{
  "conversations": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"}
  ],
  "label": true
}
```

### 2. Configure Training

Edit `config/config.yaml` to customize:

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
  max_seq_length: 2048

lora:
  rank: 16
  alpha: 32
  dropout: 0.05

training:
  num_epochs: 1
  per_device_batch_size: 2
  learning_rate: 1.0e-4
  gradient_accumulation_steps: 4

data:
  dataset_path: "syngen_toolset_v1.0.0_claude.jsonl"
  train_split: 0.8
```

### 3. Run Training

```bash
python main.py --config config/config.yaml
```

### 4. Monitor Progress

Training logs are written to:
- Console: Real-time progress
- `logs/training.log`: Human-readable log file
- `logs/training.jsonl`: Structured JSON logs for analysis

### 5. Resume from Checkpoint

```bash
python main.py --config config/config.yaml --resume checkpoints/checkpoint_step_500.npz
```

### 6. Run Evaluation Only

```bash
python main.py --config config/config.yaml --resume checkpoints/best_checkpoint.npz --eval-only
```

## Project Structure

```
mistral_lora_mac/
├── config/
│   ├── __init__.py
│   ├── config.yaml              # Main configuration file
│   └── config_manager.py        # Configuration management
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── __init__.py
│   ├── data/
│   │   └── __init__.py
│   ├── model/
│   │   └── __init__.py
│   ├── training/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   ├── data_pipeline.py         # Data loading and preprocessing
│   ├── model_manager.py         # Model loading and LoRA application
│   ├── trainer.py               # Training engine
│   ├── evaluator.py             # Evaluation and inference
│   └── utils.py                 # Logging, monitoring, utilities
├── logs/                        # Training logs (auto-created)
├── checkpoints/                 # Model checkpoints (auto-created)
├── outputs/                     # Final models and metrics (auto-created)
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Architecture

### Core Modules

1. **Configuration Manager** (`config/config_manager.py`)
   - Loads and validates YAML configuration
   - Supports environment variable overrides
   - Type-safe configuration objects

2. **Data Pipeline** (`src/data_pipeline.py`)
   - JSONL dataset loading and validation
   - Mistral Instruct format conversion
   - Tokenization with padding/truncation
   - Train/validation splitting with stratification

3. **Model Manager** (`src/model_manager.py`)
   - Mistral-7B model loading
   - LoRA adapter application
   - Parameter freezing and management
   - Adapter save/load

4. **Training Engine** (`src/trainer.py`)
   - Training loop with gradient accumulation
   - AdamW optimizer with cosine warmup scheduler
   - Gradient clipping and monitoring
   - Checkpoint management

5. **Evaluator** (`src/evaluator.py`)
   - Validation loss computation
   - Perplexity calculation
   - Text generation for qualitative assessment

6. **Utilities** (`src/utils.py`)
   - Structured logging
   - Memory monitoring (RAM + Metal GPU)
   - Device detection
   - Helper functions

## Configuration Reference

### Model Configuration

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"  # HuggingFace model ID
  cache_dir: "~/.cache/huggingface"            # Model cache directory
  dtype: "float16"                              # Data type (float16/float32)
  max_seq_length: 2048                          # Maximum sequence length
```

### LoRA Configuration

```yaml
lora:
  rank: 16                      # LoRA rank (higher = more capacity, more memory)
  alpha: 32                     # LoRA alpha (typically 2 * rank)
  dropout: 0.05                 # Dropout for LoRA layers
  target_modules:               # Which modules to apply LoRA to
    - "q_proj"                  # Query projection
    - "v_proj"                  # Value projection
```

### Training Configuration

```yaml
training:
  num_epochs: 1                          # Number of training epochs
  per_device_batch_size: 2               # Batch size per device
  gradient_accumulation_steps: 4         # Accumulation steps (effective batch = 8)
  learning_rate: 1.0e-4                  # Peak learning rate
  warmup_steps: 100                      # Warmup steps for LR scheduler
  max_grad_norm: 1.0                     # Gradient clipping threshold
  weight_decay: 0.01                     # Weight decay for AdamW
  save_steps: 100                        # Save checkpoint every N steps
  eval_steps: 50                         # Evaluate every N steps
  logging_steps: 10                      # Log metrics every N steps
```

### Data Configuration

```yaml
data:
  dataset_path: "syngen_toolset_v1.0.0_claude.jsonl"  # Path to JSONL file
  train_split: 0.8                                     # Train/validation split ratio
  shuffle: true                                        # Shuffle data
  seed: 42                                             # Random seed
```

## Expected Performance

On Mac M4 with 16GB RAM:

- **Training Time**: ~4-6 hours for 1 epoch (1000 examples)
- **Peak Memory**: ~14-16 GB (RAM + Metal)
- **Throughput**: ~0.3-0.5 steps/second
- **Trainable Parameters**: ~8-10M (< 1% of 7B)

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `per_device_batch_size: 1`
2. Reduce sequence length: `max_seq_length: 1024`
3. Reduce LoRA rank: `rank: 8`
4. Increase gradient accumulation: `gradient_accumulation_steps: 8`

### Metal GPU Not Available

```bash
# Check MLX installation
pip install --upgrade mlx

# Verify Metal
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

### Dataset Loading Errors

- Verify JSONL format: each line must be valid JSON
- Check required fields: `conversations` and `label`
- Validate conversation format: `role` and `content` fields

### Model Download Issues

```bash
# Set HuggingFace cache directory
export HF_HOME="/path/to/cache"

# Login to HuggingFace (if model requires authentication)
huggingface-cli login
```

## Advanced Usage

### Custom Dataset

Create your own JSONL dataset:

```python
import json

data = {
    "conversations": [
        {"role": "user", "content": "Your question"},
        {"role": "assistant", "content": "Model's answer"}
    ],
    "label": True  # True for desirable, False for undesirable
}

with open("my_dataset.jsonl", "a") as f:
    f.write(json.dumps(data) + "\n")
```

### Environment Variable Overrides

```bash
# Override configuration via environment variables
export LEARNING_RATE=2e-4
export BATCH_SIZE=1
export NUM_EPOCHS=3
export DATASET_PATH="/path/to/dataset.jsonl"

python main.py
```

### Hyperparameter Tuning

Create experiment configs:

```bash
# config/experiment_configs/high_lr.yaml
training:
  learning_rate: 5.0e-4

# Run experiment
python main.py --config config/experiment_configs/high_lr.yaml
```

## Output Files

After training:

```
outputs/
├── final_model/
│   └── lora_adapters.npz         # Final LoRA weights
├── metrics/
│   └── training_report.json      # Training summary
checkpoints/
├── checkpoint_step_100.npz       # Checkpoint at step 100
├── checkpoint_step_200.npz       # Checkpoint at step 200
├── best_checkpoint.npz           # Best validation loss checkpoint
logs/
├── training.log                  # Human-readable logs
└── training.jsonl                # Structured JSON logs
```

## Inference with Fine-Tuned Model

```python
from src.model_manager import ModelManager
from src.evaluator import Evaluator

# Load configuration
# ... (see main.py for full example)

# Load model and adapters
model_manager = ModelManager(config, logger)
model = model_manager.load_base_model()
model = model_manager.apply_lora()
model_manager.load_adapters("outputs/final_model/lora_adapters.npz")

# Create evaluator
evaluator = Evaluator(model, tokenizer, config, logger)

# Generate text
prompts = ["Can you help me organize my notes?"]
samples = evaluator.generate_samples(prompts, max_new_tokens=100)

for sample in samples:
    print(f"Prompt: {sample.prompt}")
    print(f"Generated: {sample.generated_text}")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mlx_finetuning_2024,
  title = {MLX Fine-Tuning System for Mistral-7B},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/mlx-finetuning}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- MLX framework by Apple
- Mistral AI for Mistral-7B-Instruct-v0.3
- LoRA paper: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
- HuggingFace Transformers

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/yourusername/mlx-finetuning/issues
- Documentation: See `docs/` folder for detailed guides
- Contact: your.email@example.com

## Roadmap

- [ ] Multi-GPU support (when MLX supports it)
- [ ] QLoRA implementation for 4-bit quantization
- [ ] Integration with Weights & Biases
- [ ] Additional base models (Llama, Phi, etc.)
- [ ] Model merging for deployment
- [ ] Gradio/Streamlit UI
