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
└── local-training/
    ├── QUICK_REFERENCE.txt ⭐ Start here!
    ├── LOCAL_TRAINING_SETUP.md
    ├── 00-preparation-summary.md
    ├── mac-m4-kto-finetuning.md
    ├── rtx3070-kto-finetuning.md
    └── platform-comparison-analysis.md
```

## 📚 Quick Navigation

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

**Q: Which platform should I use?**
A: For KTO specifically → RTX 3070. For general fine-tuning → Mac M4 (faster with LoRA).

**Q: Can I run KTO on my Mac?**
A: PyTorch+MPS technically yes, but slow. MLX LoRA is recommended instead (similar benefits).

**Q: How long does training take?**
A: ~5-8 hours for 1000 examples on RTX 3070, ~4-6 hours on Mac M4 with MLX.

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
