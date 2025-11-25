# Nebius AI Cloud - Quick Start Guide

This is the **fast-track guide** to running your Toolset-Training pipeline on Nebius AI Cloud. Choose your preferred approach below.

## Prerequisites

1. **Nebius Account** - Sign up at [nebius.com](https://nebius.com/)
2. **API Key** - Generate from Nebius console
3. **Repository Ready** - Your Toolset-Training repo

---

## 🚀 Option 1: JupyterHub (Fastest Start - 10 minutes)

**Best for:** Testing, experimentation, interactive development

### Steps

1. **Deploy JupyterHub**
   - Go to [Nebius JupyterHub](https://nebius.com/third-party-applications/jupyter-hub)
   - Select: H100 GPU (1 or 8 GPUs)
   - Click "Deploy"
   - Wait 2-3 minutes for provisioning

2. **Upload Files**
   ```bash
   # From your local machine
   cd ~/Toolset-Training
   tar -czf toolset.tar.gz Trainers/ Datasets/ Tools/ docs/

   # Upload via JupyterHub UI or:
   scp toolset.tar.gz <jupyter-host>:/workspace/
   ```

3. **Extract and Run**
   - Open JupyterHub in browser
   - Upload `docs/nebius_training_notebook.ipynb`
   - Run all cells!

**Cost:** ~$1.13 for full SFT training (45 min at $1.50/hour Explorer rate)

### Pre-configured Notebook

Use `docs/nebius_training_notebook.ipynb` - it includes:
- ✅ Environment setup
- ✅ SFT training
- ✅ Model testing
- ✅ HuggingFace upload
- ✅ Cost estimates

---

## 🔧 Option 2: Compute VM (Production - 30 minutes)

**Best for:** Long training runs, automation, production pipelines

### Steps

1. **Create VM**

   Via Nebius Console:
   - Navigate to Compute → Create Instance
   - Select: Ubuntu 22.04 AI/ML image
   - GPU: 8x H100
   - Storage: 500GB SSD
   - Add your SSH key
   - Create

2. **SSH and Setup**
   ```bash
   # SSH to VM
   ssh ubuntu@<vm-ip>

   # Clone repository
   git clone https://github.com/ProfSynapse/Toolset-Training.git
   cd Toolset-Training

   # Run setup
   cd Trainers/rtx3090_sft
   bash setup.sh

   # Verify GPU
   nvidia-smi
   ```

3. **Train**
   ```bash
   # SFT training
   ./train.sh --model-size 7b --wandb --wandb-project nebius-training

   # Or KTO training
   cd ../rtx3090_kto
   ./train.sh --model-size 7b
   ```

4. **Run in Background (Optional)**
   ```bash
   # Start in tmux session
   tmux new-session -s training
   cd ~/Toolset-Training/Trainers/rtx3090_sft
   ./train.sh --model-size 7b

   # Detach: Ctrl+B then D
   # Reattach: tmux attach -t training
   ```

5. **Download Results**
   ```bash
   # From your local machine
   scp -r ubuntu@<vm-ip>:~/Toolset-Training/Trainers/rtx3090_sft/sft_output_rtx3090/ ./
   ```

**Cost:** ~$1.50 for full SFT training (45 min) + ~$0.50 for KTO (15 min) = **~$2.00 total**

---

## ☁️ Option 3: SkyPilot (Advanced - 1 hour)

**Best for:** Multi-node training, cost optimization, infrastructure-as-code

### Steps

1. **Install SkyPilot**
   ```bash
   pip install "skypilot-nightly[nebius]"
   sky check nebius
   ```

2. **Configure Credentials**
   ```bash
   # Follow prompts to add Nebius API key
   sky check nebius
   ```

3. **Launch Training**
   ```bash
   cd ~/Toolset-Training

   # Launch using provided config
   sky launch docs/nebius_skypilot_config.yaml
   ```

4. **Monitor**
   ```bash
   # View logs
   sky logs toolset-training --follow

   # SSH to instance
   sky ssh toolset-training

   # Check status
   sky status
   ```

5. **Stop When Done**
   ```bash
   sky down toolset-training
   ```

**Cost:** Same as VM option, but can use spot instances for ~50% savings

**Spot Instances:**
```bash
sky launch --use-spot docs/nebius_skypilot_config.yaml
```

---

## 💰 Cost Comparison

| Training Type | Duration | Explorer ($1.50/h) | On-Demand ($2.00/h) | 12-mo Reserve ($3.15/h) |
|---------------|----------|-------------------|-------------------|----------------------|
| SFT (7B) | 45 min | **$1.13** | $1.50 | $2.36 |
| KTO (7B) | 15 min | **$0.38** | $0.50 | $0.79 |
| Full Pipeline | 60 min | **$1.50** | $2.00 | $3.15 |

**Explorer Tier Benefits:**
- $1.50/hour for first 1,000 hours/month
- Available until March 2025
- Perfect for development and experimentation
- **~100+ training runs for <$100/month**

---

## ⚡ Performance Comparison

| Hardware | SFT Training Time | KTO Training Time | Cost (Explorer) |
|----------|------------------|-------------------|----------------|
| RTX 3090 (24GB) | ~45 min | ~15 min | (Local) |
| H100 (80GB) | **~15 min** | **~5 min** | $0.38 + $0.13 = **$0.51** |

**H100 Benefits:**
- 3x faster training
- 3.3x more VRAM (80GB vs 24GB)
- Can increase batch sizes
- Better for larger models (13B, 20B)

---

## 🎯 Recommended Workflow

### For First-Time Users (Testing)
1. **Start with JupyterHub**
   - Use `nebius_training_notebook.ipynb`
   - Run 1 epoch of SFT (~15 min)
   - Cost: ~$0.38
   - Validates everything works

### For Development (Iteration)
2. **Switch to Compute VM**
   - Setup once, reuse many times
   - Run full training pipeline
   - Use tmux for long runs
   - Cost: ~$2-5 per full pipeline

### For Production (Scale)
3. **Adopt SkyPilot**
   - Infrastructure-as-code
   - Multi-node if needed
   - Spot instances for savings
   - Automated recovery

---

## 📊 Quick Wins

### 1. Single Training Run (30 min total)
```bash
# Use VM approach
ssh ubuntu@<vm-ip>
git clone <repo>
cd Toolset-Training/Trainers/rtx3090_sft
bash setup.sh --quick
./train.sh --model-size 7b
```
**Result:** Trained 7B model in ~15 min
**Cost:** ~$0.38 (Explorer tier)

### 2. Full Pipeline + Upload (1 hour total)
```bash
# SFT + KTO + Upload
cd Trainers/rtx3090_sft
./train.sh --model-size 7b
./upload_model.sh  # Interactive upload

cd ../rtx3090_kto
./train.sh --model-size 7b
./upload_model.sh
```
**Result:** SFT model + KTO refined model on HuggingFace
**Cost:** ~$1.50 (Explorer tier)

### 3. Experiment Suite (2-3 hours)
```bash
# Try different model sizes
./train.sh --model-size 3b   # Fast iteration (~5 min)
./train.sh --model-size 7b   # Production (~15 min)
./train.sh --model-size 13b  # Max quality (~30 min)
```
**Result:** 3 models with different tradeoffs
**Cost:** ~$4-5 (Explorer tier)

---

## 🔍 Monitoring and Debugging

### Check GPU Usage
```bash
# On VM or SSH'd into SkyPilot
nvidia-smi -l 1  # Update every second
```

### View Training Logs
```bash
# Real-time monitoring
tail -f sft_output_rtx3090/*/logs/training_latest.jsonl

# Or in notebook
!tail -f /workspace/Trainers/rtx3090_sft/sft_output_rtx3090/*/logs/training_latest.jsonl
```

### W&B Integration (Recommended)
```bash
# Set in .env file
echo "WANDB_API_KEY=your_key_here" >> .env

# Train with W&B
./train.sh --model-size 7b --wandb --wandb-project nebius-experiments
```
Then view metrics at [wandb.ai](https://wandb.ai)

---

## 🆘 Troubleshooting

### "CUDA out of memory"
- Your configs are optimized for 24GB (RTX 3090)
- H100 has 80GB - you can **increase** batch sizes!
- Try: `--batch-size 12` instead of default 6

### "Training seems slow"
- Check GPU utilization: `nvidia-smi`
- Should be >90% GPU utilization
- If low, increase batch size

### "Can't connect to VM"
- Check security groups (SSH port 22 open)
- Verify SSH key was added correctly
- Try: `ssh -v ubuntu@<vm-ip>` for verbose output

### "Dataset not found"
- Verify files uploaded: `ls -lh Datasets/`
- Check paths in config match uploaded location
- JupyterHub: Use `/workspace/Datasets/...`
- VM: Use `~/Toolset-Training/Datasets/...`

---

## 📚 Next Steps

After your first successful training:

1. **Experiment with hyperparameters**
   - Learning rates: `--learning-rate 1e-4` to `5e-4`
   - Batch sizes: `--batch-size 8` to `--batch-size 16`
   - Epochs: `--num-epochs 2` to `--num-epochs 5`

2. **Try different model sizes**
   - 3B for fast iteration
   - 7B for production (recommended)
   - 13B for maximum quality
   - 20B for specialized tasks

3. **Integrate into your workflow**
   - Add to CI/CD pipeline
   - Schedule periodic retraining
   - A/B test different configurations

4. **Scale up**
   - Multi-node training with SkyPilot
   - Distributed data parallel
   - Experiment tracking with W&B

---

## 🎓 Resources

- **Full Integration Guide:** `docs/nebius-integration-guide.md`
- **Jupyter Notebook Template:** `docs/nebius_training_notebook.ipynb`
- **SkyPilot Config:** `docs/nebius_skypilot_config.yaml`
- **Nebius Docs:** [docs.nebius.com](https://docs.nebius.com/)
- **SkyPilot Docs:** [docs.skypilot.co](https://docs.skypilot.co/)

---

## Summary

| Approach | Setup Time | Cost per Run | Best For |
|----------|-----------|-------------|----------|
| **JupyterHub** | 10 min | $0.38-1.50 | Testing, experimentation |
| **Compute VM** | 30 min | $0.38-1.50 | Production, automation |
| **SkyPilot** | 1 hour | $0.20-1.50 | Scale, multi-node, cost optimization |

**Recommended First Step:** Try JupyterHub with the provided notebook for immediate results!

**Questions?** Check `docs/nebius-integration-guide.md` for comprehensive documentation.

---

**Happy Training! 🚀**
