# Synthetic Conversations Documentation Index

Complete documentation for the Claudesidian synthetic dataset and KTO fine-tuning project.

## 📁 Folder Structure

```
docs/
├── INDEX.md (this file)
├── WORKSPACE_README.md
├── WORKSPACE_ANALYSIS_REPORT.md
├── WORKSPACE_ARCHITECTURE_DIAGRAM.md
├── WORKSPACE_DOCUMENTATION_INDEX.md
├── WORKSPACE_KEY_FILES_REFERENCE.md
├── local-training/
│   ├── QUICK_REFERENCE.txt ⭐ Start here for local!
│   ├── LOCAL_TRAINING_SETUP.md
│   ├── 00-preparation-summary.md
│   ├── mac-m4-kto-finetuning.md
│   ├── rtx3070-kto-finetuning.md
│   └── platform-comparison-analysis.md
└── cloud-training/
    ├── NEBIUS_QUICKSTART.md ⭐ Start here for cloud!
    ├── NEBIUS_INTEGRATION_SUMMARY.md
    ├── nebius-integration-guide.md
    ├── nebius_training_notebook.ipynb
    └── nebius_skypilot_config.yaml
```

## 📚 Quick Navigation

### For Cloud GPU Training (Nebius AI) 🆕

**Start with**: `NEBIUS_QUICKSTART.md` - 10-minute setup guide

Then choose your approach:
- **JupyterHub** - Interactive notebooks, fastest start (10 min)
- **Compute VM** - Production training, full control (30 min)
- **SkyPilot** - Multi-node orchestration, cost optimization (1 hour)

**Cost:** $0.50-1.50 per full training run (3x faster than local RTX 3090)

### For Local Hardware Training Setup

**Start with**: `local-training/QUICK_REFERENCE.txt`

Then read based on your hardware:
- **Mac M4 (24GB)**: `local-training/mac-m4-kto-finetuning.md`
- **NVIDIA RTX 3070 (8GB)**: `local-training/rtx3070-kto-finetuning.md`

### For Project Overview

- `WORKSPACE_README.md` - Project overview and structure
- `WORKSPACE_ANALYSIS_REPORT.md` - Detailed analysis
- `WORKSPACE_ARCHITECTURE_DIAGRAM.md` - System architecture
- `WORKSPACE_KEY_FILES_REFERENCE.md` - Important files reference
- `WORKSPACE_DOCUMENTATION_INDEX.md` - All documentation

## ☁️ Cloud Training Documentation (Nebius AI) 🆕

### Why Use Nebius?
- ✅ **3x faster** than local RTX 3090 (H100 GPUs)
- ✅ **No code changes** - existing scripts work as-is
- ✅ **Cost-effective** - $0.50-1.50 per full SFT+KTO pipeline
- ✅ **Explorer Tier** - $1.50/GPU-hour for first 1,000 hours/month
- ✅ **80GB VRAM** - vs 24GB local (run larger models, bigger batches)

### Integration Approaches

| Approach | Setup | Cost/Run | Best For |
|----------|-------|----------|----------|
| **JupyterHub** | 10 min | $0.38-1.50 | Testing, experimentation |
| **Compute VM** | 30 min | $0.38-1.50 | Production, automation |
| **SkyPilot** | 1 hour | $0.20-1.50 | Multi-node, cost optimization |

### Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `NEBIUS_QUICKSTART.md` | Fast-track guide to get started | Everyone |
| `NEBIUS_INTEGRATION_SUMMARY.md` | Research summary and recommendations | Decision makers |
| `nebius-integration-guide.md` | Comprehensive setup guide (10,000+ words) | Implementation |
| `nebius_training_notebook.ipynb` | Ready-to-use Jupyter notebook | JupyterHub users |
| `nebius_skypilot_config.yaml` | SkyPilot infrastructure-as-code | Advanced users |

### Quick Start

1. **Sign up** at [nebius.com](https://nebius.com/)
2. **Read** `NEBIUS_QUICKSTART.md` (10 min)
3. **Deploy** JupyterHub with H100 GPU
4. **Upload** `nebius_training_notebook.ipynb`
5. **Run** training (15 min for 7B SFT)
6. **Cost:** ~$0.38 for first test run

### Performance Comparison

| Hardware | SFT (7B) | KTO (7B) | VRAM | Cost |
|----------|----------|----------|------|------|
| RTX 3090 (local) | 45 min | 15 min | 24GB | Free (power) |
| **H100 (Nebius)** | **15 min** | **5 min** | **80GB** | **$0.38-0.50** |

## 🎯 Local Training Documentation

### Recommended Setups

**Mac M4 (24GB)**
- **Best**: MLX Framework with LoRA
  - 12-15 tokens/sec on 7B models
  - File: `local-training/mac-m4-kto-finetuning.md`

- **Alternative**: PyTorch + MPS + KTO
  - Slower but has native KTO support
  - Same file, "Option 2" section

**NVIDIA RTX 3070 (8GB)**
- **Best**: Unsloth + TRL (KTO)
  - 10-15 tokens/sec on 7B models
  - File: `local-training/rtx3070-kto-finetuning.md`

### Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `QUICK_REFERENCE.txt` | Quick start guide and key findings | Everyone |
| `LOCAL_TRAINING_SETUP.md` | Navigation and quick start steps | Everyone |
| `00-preparation-summary.md` | Full research summary (~200 sections) | Decision makers |
| `mac-m4-kto-finetuning.md` | Complete Mac M4 setup guide | Mac users |
| `rtx3070-kto-finetuning.md` | Complete RTX 3070 setup guide | NVIDIA users |
| `platform-comparison-analysis.md` | Side-by-side comparison | Comparing platforms |

## 🚀 Getting Started

### Step 1: Choose Your Platform
- Do you have a Mac M4 (24GB)?
- Or NVIDIA RTX 3070 (8GB)?

### Step 2: Read Quick Reference
Open `local-training/QUICK_REFERENCE.txt` (5-10 minutes)

### Step 3: Read Platform Guide
- Mac → `local-training/mac-m4-kto-finetuning.md`
- NVIDIA → `local-training/rtx3070-kto-finetuning.md`

### Step 4: Follow Setup Instructions
Complete installation and configuration (30-45 minutes)

### Step 5: Start Training
Run the training script with your dataset

## 📊 Dataset Information

**Your Dataset**: Claudesidian Synthetic Training Dataset
- **Location**: professorsynapse/claudesidian-synthetic-dataset
- **File**: syngen_toolset_v1.0.0_claude.jsonl
- **Size**: 1.55 MB
- **Examples**: 1,000 total
  - Desirable: 746 (74.6%)
  - Undesirable: 254 (25.4%)
  - Ratio: 2.94:1

Both platform guides include code for loading and formatting this dataset.

## ⚠️ Important Limitations

### Mac M4
- ❌ KTO not natively supported in MLX (use LoRA instead)
- ❌ PyTorch MPS experimental and slow
- ✅ MLX provides similar fine-tuning benefits

### NVIDIA RTX 3070
- ❌ 8GB VRAM = max 7B models
- ❌ Windows/Linux only
- ✅ Full KTO support via Unsloth + TRL

## 🔗 Related Files in Project

**Dataset**:
- `syngen_toolset_v1.0.0_claude.jsonl` - Your 1000 examples

**Notebooks**:
- `kto_colab_notebook.ipynb` - Colab/GPU training notebook

**References**:
- `TOOL_SCHEMA_REFERENCE.md` - Tool definitions
- `SCHEMA_VERIFICATION_REFERENCE.md` - Validation info
- `finetuning-strategy.md` - Original strategy document
- `README.md` - Project readme

## 💡 Common Questions

**Q: Should I use cloud (Nebius) or local training?**
A:
- **Cloud (Nebius):** 3x faster, no hardware needed, ~$0.50-1.50 per run
- **Local:** Free after hardware cost, good for learning/experimentation
- **Recommendation:** Try Nebius JupyterHub first ($0.38 test), then decide

**Q: Which platform should I use for local training?**
A: For KTO specifically → RTX 3070. For general fine-tuning → Mac M4 (faster with LoRA).

**Q: Can I use my existing training scripts on Nebius?**
A: Yes! No code changes needed. Your `train.sh` scripts work as-is on Nebius VMs.

**Q: How much does Nebius cost?**
A: Explorer tier = $1.50/GPU-hour (first 1,000 hours/month). Full SFT+KTO pipeline = $0.50-1.50.

**Q: Can I run KTO on my Mac?**
A: PyTorch+MPS technically yes, but slow. MLX LoRA is recommended instead (similar benefits).

**Q: How long does training take?**
A:
- **Nebius H100:** SFT ~15 min, KTO ~5 min (7B model)
- **RTX 3090 (local):** SFT ~45 min, KTO ~15 min (7B model)
- **RTX 3070:** ~5-8 hours for 1000 examples
- **Mac M4 (MLX):** ~4-6 hours for 1000 examples

**Q: What model should I start with?**
A: 3B parameter models for testing, 7B for production. See platform guide for recommendations.

## 📞 Support

All documentation includes:
- ✅ Detailed installation steps
- ✅ Configuration examples
- ✅ Troubleshooting guides
- ✅ Performance optimization tips
- ✅ Common error solutions

See specific sections in platform guides for help.

## 📝 Documentation Versions

- **Created**: November 9, 2025
- **Snapshot**: January 2025 research (latest versions as of that date)
- **Updated**: During this session

All guides are current and include latest best practices for the specified platforms.

---

**Ready to get started?** Open `local-training/QUICK_REFERENCE.txt` 🚀
