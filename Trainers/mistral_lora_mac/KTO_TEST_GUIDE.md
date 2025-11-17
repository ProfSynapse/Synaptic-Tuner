# KTO Training Test Guide

## Overview

This is a lightweight test configuration to validate the **first KTO (Kahneman-Tversky Optimization) implementation for Apple's MLX framework**.

## Test Configuration

### Resources Required
- **Time**: 5-10 minutes
- **Memory**: ~6-8 GB RAM
- **GPU**: Apple Silicon (M1/M2/M3/M4)
- **Dataset**: 100 examples (50 desirable, 50 undesirable)

### Test Settings
```yaml
Model:
  - Sequence length: 512 tokens (vs 4096 full)
  - LoRA rank: 8 (vs 64 full)
  - Target modules: q_proj, v_proj only

Training:
  - Max steps: 50
  - Batch size: 2 (effective: 4)
  - Learning rate: 5e-5
  - Epochs: 1

KTO:
  - Beta: 0.1
  - Lambda D: 1.0
  - Lambda U: 1.0
```

## How to Run

### Quick Start
```bash
cd "/Users/jrosenbaum/Documents/Code/Synthetic Conversations/code/mistral_lora_mac"
source venv/bin/activate
./test_kto.sh
```

### Manual Run
```bash
python main.py --config config/config_test_kto.yaml
```

## What to Look For

### ✅ Success Indicators

1. **Model Loading**
   ```
   ✓ MLX is installed
   ✓ Metal GPU is available and working
   ✓ Model loaded successfully
   ```

2. **Reference Model Creation**
   ```
   Creating frozen reference model for KTO...
   Reference model created and frozen
   ```

3. **KTO Training Mode**
   ```
   Initializing Trainer (KTO Mode)
   Training mode: KTO (Kahneman-Tversky Optimization)
   KTO beta: 0.1
   KTO lambda_d: 1.0
   KTO lambda_u: 1.0
   ```

4. **Loss Computation**
   - Loss should start around 2-4 (typical for cross-entropy)
   - Loss should gradually decrease
   - No NaN or Inf values

5. **Batch Processing**
   - Should see batches with both desirable and undesirable examples
   - Warning messages if batch has only one type (normal, uses fallback)

### ❌ Potential Issues

1. **Memory Errors**
   - Reduce `per_device_batch_size` to 1
   - Reduce `max_seq_length` to 256

2. **Model Loading Fails**
   - Check internet connection (downloads model)
   - Check HuggingFace cache: `~/.cache/huggingface`

3. **Configuration Errors**
   - Verify config loads: `python -c "from config.config_manager import ConfigurationManager; ConfigurationManager().load('config/config_test_kto.yaml')"`

## Output Files

After successful run:
```
test_logs/
  ├── training.log          # Human-readable logs
  └── training.jsonl        # Structured logs

test_checkpoints/
  ├── checkpoint_step_20.npz
  ├── checkpoint_step_40.npz
  └── best_checkpoint.npz

test_outputs/
  └── final_model/
      └── lora_adapters.npz  # Final trained model
```

## Validation Steps

### 1. Check Logs
```bash
tail -50 test_logs/training.log
```

Look for:
- "Training complete"
- Final loss values
- No error messages

### 2. Verify KTO Loss Computation
```bash
grep "KTO" test_logs/training.log
```

Should show:
- KTO parameters loaded
- KTO loss being used
- Both chosen and rejected examples processed

### 3. Check Model Outputs
```bash
ls -lh test_outputs/final_model/
```

Should contain `lora_adapters.npz` file

## Performance Expectations

### Mac M4 (24GB RAM)
- **Training speed**: ~0.3-0.5 steps/sec
- **Total time**: ~2-3 minutes for 50 steps
- **Peak memory**: ~8-10 GB

### Mac M2/M3 (16GB RAM)
- **Training speed**: ~0.2-0.4 steps/sec
- **Total time**: ~3-5 minutes for 50 steps
- **Peak memory**: ~6-8 GB

### Mac M1 (8-16GB RAM)
- **Training speed**: ~0.1-0.3 steps/sec
- **Total time**: ~5-10 minutes for 50 steps
- **Peak memory**: ~6-8 GB

## Next Steps After Test

### If Test Passes ✅
1. Run full training with `config/config.yaml`
2. Monitor for several hours
3. Evaluate model quality

### If Test Fails ❌
1. Check error messages in `test_logs/training.log`
2. Verify configuration loads
3. Check system resources (memory, disk space)
4. Review implementation for bugs

## Technical Details

### KTO Implementation
- **Reference model**: Frozen copy of initial model
- **Loss function**: Kahneman-Tversky value function using sigmoid
- **Batch processing**: Separates desirable/undesirable examples
- **Fallback**: Uses cross-entropy if batch is homogeneous

### MLX Optimizations
- Lazy evaluation for memory efficiency
- Metal GPU acceleration
- Gradient checkpointing support
- Efficient parameter freezing

## Support

Issues? Check:
1. System requirements met
2. All dependencies installed
3. Dataset format correct
4. Configuration valid

For bugs or questions, see main README.
