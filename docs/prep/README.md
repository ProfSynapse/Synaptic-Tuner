# Preparation Phase Documentation

Complete research and preparation documentation for KTO fine-tuning on local hardware.

## 📁 Contents

### local-training/
Comprehensive guides for running fine-tuning locally:

- **QUICK_REFERENCE.txt** - Quick start overview
- **LOCAL_TRAINING_SETUP.md** - Navigation guide
- **00-preparation-summary.md** - Full research summary (~200 sections)

#### Platform-Specific Guides
- **mac-m4-kto-finetuning.md** - Original Mac M4 guide (generic KTO)
- **mac-m4-mistral-7b-setup.md** ⭐ **UPDATED** - Mac M4 with Mistral 7B v0.3 + local JSONL dataset
- **rtx3070-kto-finetuning.md** - NVIDIA RTX 3070 guide (Unsloth + TRL)
- **platform-comparison-analysis.md** - Side-by-side comparison

## 🎯 For Mac M4 Users (Recommended)

**Use**: `local-training/mac-m4-mistral-7b-setup.md`

This updated guide includes:
- ✅ Mistral-7B-Instruct-v0.3 model configuration
- ✅ Local JSONL dataset loading (syngen_toolset_v1.0.0_claude.jsonl)
- ✅ MLX framework setup (2-3x faster than PyTorch MPS)
- ✅ LoRA fine-tuning (parameter-efficient training)
- ✅ Expected 4-6 hour training time on 24GB M4
- ✅ Complete training script
- ✅ Troubleshooting guide

### Key Features of Updated Guide
- Model: Mistral-7B-Instruct-v0.3 (chosen for balance of performance & efficiency)
- Dataset: Your local syngen_toolset_v1.0.0_claude.jsonl (1000 examples)
- Framework: MLX (Apple's native ML framework)
- Method: LoRA (low-rank adaptation for efficient fine-tuning)
- Training Time: ~4-6 hours
- Memory: ~14-16GB peak

## 🎯 For NVIDIA RTX 3070 Users

**Use**: `local-training/rtx3070-kto-finetuning.md`

This guide includes:
- ✅ Unsloth + TRL setup (native KTO support)
- ✅ CUDA configuration
- ✅ 8GB VRAM optimization techniques
- ✅ Full KTO training

## 📊 Research Summary

**00-preparation-summary.md** contains:
- Comprehensive platform comparison
- Technology analysis (MLX, Unsloth, PyTorch, TRL)
- Performance benchmarks
- Hardware requirements
- Installation verification steps
- Troubleshooting guides

---

## 🚀 Next Phase: Architecture

After reviewing this preparation documentation, the Architect phase will:

1. Design the complete training pipeline
2. Define component interactions
3. Create architecture diagrams
4. Specify API/data interfaces
5. Document design decisions
6. Plan implementation structure

See the **architecture/** folder (created during Architecture phase) for design specifications.

---

## 📋 Key Decisions Made in Preparation Phase

✅ **Model Choice**: Mistral-7B-Instruct-v0.3
- Strong reasoning capabilities for tool use tasks
- Excellent balance of performance and efficiency
- Well-optimized for both M4 and RTX 3070
- Good community support and documentation

✅ **Framework for Mac**: MLX
- Native Apple Silicon support
- 2-3x faster than PyTorch MPS
- Cleaner API and better M4 memory management
- Proven performance on similar tasks

✅ **Method**: LoRA
- 95-99% parameter efficiency vs full fine-tuning
- Faster training and lower memory
- Similar convergence and quality to KTO for this use case
- Well-supported across all frameworks

✅ **Dataset**: Local JSONL
- Use your existing syngen_toolset_v1.0.0_claude.jsonl
- 1000 examples (746 desirable, 254 undesirable)
- Matched ratio for effective learning
- Domain-specific to Claudesidian tool use

---

## 📞 Questions?

Refer to the specific platform guide:
- Mac M4 → `local-training/mac-m4-mistral-7b-setup.md`
- RTX 3070 → `local-training/rtx3070-kto-finetuning.md`

Both include:
- Installation verification steps
- Detailed configuration examples
- Performance expectations
- Complete troubleshooting guides
- Expected training metrics

---

**Status**: Preparation Phase Complete ✅

**Next**: Architecture Phase (See architecture/ folder when ready)
