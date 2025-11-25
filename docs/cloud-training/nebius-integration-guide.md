# Nebius AI Cloud Integration Guide

This guide outlines different approaches for running the Toolset-Training pipeline on Nebius AI Cloud infrastructure.

## Overview

Nebius AI Cloud is a GPU cloud platform optimized for AI/ML workloads, offering NVIDIA H100, H200, L40S GPUs with InfiniBand networking. This guide covers three integration approaches for running your SFT and KTO training pipelines on Nebius.

## Integration Approaches

### 1. Managed JupyterHub (Recommended for Quick Start)

**Best for:** Interactive development, experimentation, and smaller training runs.

#### Features
- Pre-configured JupyterHub with PyTorch and CUDA
- Direct GPU access from notebook kernels
- Pre-installed ML libraries and GPU drivers
- Multi-user environment support

#### Setup Steps

1. **Deploy JupyterHub Instance**
   - Navigate to [Nebius Third-Party Applications](https://nebius.com/third-party-applications/jupyter-hub)
   - Deploy JupyterHub with PyTorch and CUDA
   - Select GPU configuration (H100, H200, or L40S)

2. **Upload Training Code**
   ```bash
   # From your local machine, package the training code
   tar -czf toolset-training.tar.gz Trainers/ Datasets/ Tools/

   # Upload via JupyterHub UI or:
   scp toolset-training.tar.gz jupyter-instance:/workspace/
   ```

3. **Create Training Notebook**

   Create a new notebook in JupyterHub:

   ```python
   # Install dependencies
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

   # Import training modules
   import sys
   sys.path.append('/workspace/Trainers/rtx3090_sft')

   from configs.training_config import get_7b_config
   from src.model_loader import load_model_and_tokenizer
   from src.data_loader import prepare_dataset
   from train_sft import train_model

   # Configure and run training
   config = get_7b_config()
   config.dataset_config.local_file = '/workspace/Datasets/syngen_tools_sft_11.18.25.jsonl'

   # Train
   model, tokenizer = load_model_and_tokenizer(config.model_config, config.lora_config)
   train_dataset = prepare_dataset(config.dataset_config, tokenizer)
   train_model(model, tokenizer, train_dataset, config)
   ```

4. **Monitor Training**
   - Use notebook cells to monitor metrics
   - Access logs in real-time via `!tail -f` commands
   - Visualize metrics inline with matplotlib

#### Advantages
- ✅ Fastest to get started (pre-configured environment)
- ✅ Interactive debugging and experimentation
- ✅ Built-in visualization tools
- ✅ Easy file management via UI

#### Limitations
- ❌ Requires manual session management
- ❌ Less suitable for long-running jobs (multi-day training)
- ❌ No automatic retry/recovery

---

### 2. Compute VMs with Direct SSH Access

**Best for:** Production training runs, automated pipelines, CI/CD integration.

#### Features
- Full control over VM environment
- Bare-metal GPU performance
- InfiniBand networking for multi-GPU setups
- Custom images with pre-installed drivers

#### Setup Steps

1. **Create GPU VM**

   Via Nebius Console:
   - Navigate to [Compute](https://nebius.com/services/compute)
   - Create new VM with GPU configuration
   - Select NVIDIA H100 (8 GPUs recommended)
   - Choose AI/ML-ready image (Ubuntu 22.04 + CUDA)
   - Configure networking and SSH keys

   Or via CLI:
   ```bash
   # Install Nebius CLI (requires Python SDK)
   pip install nebius

   # Create VM (example - see docs for exact commands)
   nebius compute instance create \
     --name training-vm \
     --zone eu-north1-c \
     --gpu-type h100 \
     --gpu-count 8 \
     --image-family ai-ml-ubuntu-2204
   ```

2. **Setup Environment**

   SSH into your VM:
   ```bash
   ssh ubuntu@<vm-ip>

   # Clone repository
   git clone <your-repo-url> ~/Toolset-Training
   cd ~/Toolset-Training

   # Run setup script (existing setup.sh works!)
   cd Trainers/rtx3090_sft
   bash setup.sh

   # Verify GPU
   nvidia-smi
   ```

3. **Run Training**

   Use existing training scripts:
   ```bash
   # SFT Training
   cd ~/Toolset-Training/Trainers/rtx3090_sft
   ./train.sh --model-size 7b --wandb --wandb-project nebius-training

   # Or KTO Training
   cd ~/Toolset-Training/Trainers/rtx3090_kto
   ./train.sh --model-size 7b
   ```

4. **Monitor with tmux/screen**
   ```bash
   # Start training in detached session
   tmux new-session -d -s training 'cd ~/Toolset-Training/Trainers/rtx3090_sft && ./train.sh --model-size 7b'

   # Attach to monitor
   tmux attach -t training

   # Detach: Ctrl+B then D
   ```

#### Advantages
- ✅ Full environment control
- ✅ Works with existing scripts (no modification needed)
- ✅ Suitable for long-running jobs
- ✅ Easy CI/CD integration
- ✅ Can use tmux/screen for session persistence

#### Limitations
- ❌ Requires manual VM management
- ❌ No built-in experiment tracking (unless you use W&B)
- ❌ Need to handle VM lifecycle yourself

---

### 3. SkyPilot Orchestration (Advanced)

**Best for:** Multi-node training, automatic failover, cost optimization across clouds.

#### Features
- Automatic cluster provisioning
- Multi-node distributed training
- Spot instance support with auto-recovery
- Infrastructure-as-code workflow
- Cost optimization (automatically finds cheapest GPUs)

#### Setup Steps

1. **Install SkyPilot**
   ```bash
   pip install "skypilot-nightly[nebius]"

   # Configure Nebius credentials
   sky check nebius
   ```

2. **Create SkyPilot Task YAML**

   Create `sky_train_sft.yaml`:
   ```yaml
   name: toolset-sft-training

   resources:
     cloud: nebius
     accelerators: H100:8
     disk_size: 500  # GB

   file_mounts:
     /workspace:
       name: training-data
       mode: COPY
       source: .

   setup: |
     # Install conda
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     bash Miniconda3-latest-Linux-x86_64.sh -b

     # Setup environment
     cd /workspace/Trainers/rtx3090_sft
     bash setup.sh --quick

   run: |
     cd /workspace/Trainers/rtx3090_sft
     source ~/miniconda3/bin/activate unsloth_env

     ./train.sh --model-size 7b --wandb --wandb-project nebius-sky
   ```

3. **Launch Training**
   ```bash
   # Launch job
   sky launch sky_train_sft.yaml

   # Monitor logs
   sky logs toolset-sft-training

   # SSH to instance
   sky ssh toolset-sft-training

   # Stop when done
   sky down toolset-sft-training
   ```

4. **Multi-Node Training (Optional)**

   For distributed training across multiple nodes:
   ```yaml
   name: toolset-distributed-training

   resources:
     cloud: nebius
     accelerators: H100:8
     num_nodes: 2  # 16 GPUs total
     disk_size: 500

   setup: |
     # Same as above
     cd /workspace/Trainers/rtx3090_sft
     bash setup.sh --quick

   run: |
     # SkyPilot automatically configures distributed training
     cd /workspace/Trainers/rtx3090_sft
     source ~/miniconda3/bin/activate unsloth_env

     # Set distributed training env vars (SkyPilot provides these)
     export MASTER_ADDR=${SKYPILOT_NODE_IPS[0]}
     export MASTER_PORT=29500
     export WORLD_SIZE=${SKYPILOT_NUM_NODES}
     export NODE_RANK=${SKYPILOT_NODE_RANK}

     python train_sft.py --model-size 7b --distributed
   ```

#### Advantages
- ✅ Automatic cluster management
- ✅ Multi-node distributed training
- ✅ Spot instance support (cost savings)
- ✅ Auto-recovery from failures
- ✅ Infrastructure-as-code (reproducible)
- ✅ Multi-cloud support (can fall back to other clouds)

#### Limitations
- ❌ Steeper learning curve
- ❌ Requires YAML configuration
- ❌ More complex debugging

---

## Cost Comparison

### Nebius Pricing (as of 2025)

| GPU Type | Explorer Tier (first 1,000h/month) | On-Demand | 12-month Reserve |
|----------|-------------------------------------|-----------|------------------|
| H100 (80GB) | **$1.50/hour** | $2.00/hour | $3.15/hour |
| H200 | N/A | ~$5-6/hour | ~$4/hour |
| L40S | N/A | ~$1.50/hour | ~$1/hour |

**Training Cost Estimate:**
- 7B model SFT training: ~45 minutes = **$1.13** (Explorer) or **$1.50** (On-demand)
- 7B model KTO training: ~15 minutes = **$0.38** (Explorer) or **$0.50** (On-demand)
- **Monthly budget:** Under $100 for extensive experimentation with Explorer tier

### Cost Optimization Tips

1. **Use Explorer Tier** - $1.50/GPU-hour for first 1,000 hours (available until March 2025)
2. **Use Spot Instances** - Via SkyPilot for additional savings
3. **Auto-shutdown** - Configure VMs to stop when idle
4. **Efficient Training** - Your Unsloth setup already optimized (4-bit, LoRA)

---

## API Integration Options

If you want to trigger training programmatically:

### Option 1: Nebius Compute API (Python SDK)

```python
from nebius.compute import ComputeClient

# Create client
client = ComputeClient(api_key="your-api-key")

# Start VM
vm = client.instances.create(
    name="training-vm",
    zone="eu-north1-c",
    gpu_type="h100",
    gpu_count=8,
    image_family="ai-ml-ubuntu-2204"
)

# SSH and run training (via paramiko or similar)
import paramiko
ssh = paramiko.SSHClient()
ssh.connect(vm.ip_address, username="ubuntu", key_filename="~/.ssh/id_rsa")

# Execute training
stdin, stdout, stderr = ssh.exec_command(
    "cd ~/Toolset-Training/Trainers/rtx3090_sft && ./train.sh --model-size 7b"
)

# Monitor output
for line in stdout:
    print(line.strip())
```

### Option 2: SkyPilot Python API

```python
import sky

# Define task programmatically
task = sky.Task(
    name='toolset-training',
    setup='cd /workspace/Trainers/rtx3090_sft && bash setup.sh --quick',
    run='cd /workspace/Trainers/rtx3090_sft && ./train.sh --model-size 7b'
)

# Set resources
task.set_resources(sky.Resources(
    cloud=sky.Nebius(),
    accelerators='H100:8',
    disk_size=500
))

# Launch
sky.launch(task)
```

---

## Recommendations by Use Case

| Use Case | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| **Quick experimentation** | JupyterHub | Fastest setup, interactive |
| **Single training run** | Compute VM | Simple, full control |
| **Production pipeline** | Compute VM + tmux | Reliable, cost-effective |
| **Multi-node training** | SkyPilot | Built-in orchestration |
| **Cost optimization** | SkyPilot + Spot | Automatic failover, savings |
| **CI/CD integration** | Compute VM API | Programmable, repeatable |

---

## Next Steps

### Immediate Actions

1. **Sign up for Nebius** - [nebius.com](https://nebius.com/)
2. **Get API key** - Generate from Nebius console
3. **Test Explorer Tier** - Try $1.50/hour pricing
4. **Choose approach** - Based on your needs above

### Quick Win (30 minutes)

1. Deploy JupyterHub with H100 GPU
2. Upload your `Datasets/syngen_tools_sft_11.18.25.jsonl`
3. Create notebook with training code (see template above)
4. Run one epoch of 7B SFT training
5. Estimated cost: ~$0.75

### Production Setup (2-4 hours)

1. Create Compute VM with 8x H100
2. Clone repository and run `setup.sh`
3. Configure W&B for experiment tracking
4. Run full SFT + KTO pipeline
5. Upload model to HuggingFace
6. Estimated cost: ~$3-5 for complete pipeline

---

## Resources

### Official Documentation
- [Nebius AI Cloud Docs](https://docs.nebius.com/)
- [Compute Service](https://docs.nebius.com/compute)
- [JupyterHub](https://nebius.com/third-party-applications/jupyter-hub)
- [SkyPilot Integration](https://docs.nebius.com/3p-integrations/skypilot)

### API & SDKs
- [Nebius API GitHub](https://github.com/nebius/api)
- [Nebius Python SDK](https://pypi.org/project/nebius/)
- [AI Studio Cookbook](https://github.com/nebius/ai-studio-cookbook)

### Tutorials
- [Multi-Node Fine-Tuning with SkyPilot](https://nebius.com/blog/posts/skypilot-k8s-for-multi-node-fine-tuning)
- [LLM Fine-Tuning with MLflow](https://nebius.com/blog/posts/orchestrating-llm-fine-tuning-k8s-skypilot-mlflow)
- [Fine-Tuning with Nebius AI Studio](https://nebius.com/blog/posts/fine-tuning-llms-with-nebius-ai-studio)

### Pricing
- [NVIDIA GPU Pricing](https://nebius.com/prices)
- [Compute Pricing Details](https://docs.nebius.com/compute/resources/pricing)

---

## Troubleshooting

### Common Issues

**Issue: "CUDA out of memory" on Nebius**
- Solution: Your existing batch size configs should work. H100 has 80GB VRAM vs your RTX 3090's 24GB.
- You can actually increase batch sizes for faster training!

**Issue: "Slow data loading"**
- Solution: Nebius VMs have fast NVMe SSDs. Your existing dataloader should be faster.
- Consider increasing `dataloader_num_workers` from Windows default of 0 to 4-8.

**Issue: "SSH timeout during long training"**
- Solution: Use tmux or screen as shown in Compute VM section.
- Or use SkyPilot which handles this automatically.

**Issue: "How do I get my model back?"**
- Solution: Your existing `upload_model.sh` works! Just set HF_TOKEN in `.env`.
- Or use `scp` to download: `scp -r vm:/workspace/Trainers/rtx3090_sft/sft_output_rtx3090/ ./`

---

## Summary

Your training pipeline is **already compatible** with Nebius! The main changes needed:

1. **None for VM approach** - Your existing scripts work as-is
2. **Minor for JupyterHub** - Wrap in notebook cells
3. **YAML config for SkyPilot** - Define infrastructure-as-code

**Recommended first step:** Try JupyterHub for quick validation, then move to Compute VMs for production.

**Cost:** With Explorer tier ($1.50/hour), you can run 100+ training experiments for under $100/month.

**Performance gain:** H100 is ~3x faster than RTX 3090, so your 45-minute SFT training becomes ~15 minutes!
