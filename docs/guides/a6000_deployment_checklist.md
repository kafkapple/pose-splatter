# A6000 GPU Deployment Checklist - Pose Splatter 2D GS

**Date**: 2025-11-16
**Target**: NVIDIA A6000 (24GB+ GPU)
**Purpose**: 2D Gaussian Splatting full training deployment

---

## Pre-Deployment Checklist

### 1. Hardware Verification ✅

```bash
# Check GPU availability
nvidia-smi

# Expected output:
# - GPU Name: NVIDIA RTX A6000
# - Memory: 24GB or 48GB
# - CUDA Version: 11.8 or 12.x

# Verify GPU is not in use
nvidia-smi | grep "No running processes"
```

**Requirements**:
- [ ] GPU: NVIDIA A6000 (24GB+), RTX 4090, or A100
- [ ] CUDA: 11.8 or 12.x
- [ ] Available disk space: 50GB+ for data + outputs
- [ ] Network: Stable connection for git clone and data transfer

---

### 2. Software Environment ✅

```bash
# Check conda installation
conda --version

# Check Python version
python --version  # Should be 3.10+

# Verify git
git --version
```

**Requirements**:
- [ ] Conda installed (miniconda or anaconda)
- [ ] Python 3.10+ available
- [ ] Git installed
- [ ] SSH access to server (if remote)

---

## Deployment Steps

### Step 1: Clone Repository ✅

```bash
cd ~/dev  # or your preferred directory
git clone https://github.com/YOUR_USERNAME/pose-splatter.git
cd pose-splatter

# Verify clone
ls -la
# Should see: README.md, src/, scripts/, configs/, docs/
```

**Checklist**:
- [ ] Repository cloned successfully
- [ ] All directories present (src, scripts, configs, docs)
- [ ] README.md exists and is readable

---

### Step 2: Create Conda Environment ✅

```bash
# Create new environment
conda create -n splatter python=3.10 -y

# Activate environment
conda activate splatter

# Verify activation
which python
# Should show: /path/to/conda/envs/splatter/bin/python

# Install PyTorch with CUDA
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional dependencies
pip install gsplat torchmetrics matplotlib seaborn pandas tabulate h5py zarr einops

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected output**:
```
CUDA: True
GPU: NVIDIA RTX A6000
```

**Checklist**:
- [ ] Conda environment created
- [ ] Python 3.10+ installed
- [ ] PyTorch 2.0+ with CUDA installed
- [ ] CUDA availability confirmed
- [ ] GPU detected correctly
- [ ] All packages installed without errors

---

### Step 3: Data Preparation ✅

```bash
# Create data directory
mkdir -p data/markerless_mouse_1_nerf

# Option 1: Download from repository
# Follow instructions at: https://github.com/tqxli/dannce-pytorch

# Option 2: Transfer from existing location
# rsync -avz --progress user@source:/path/to/data/ data/markerless_mouse_1_nerf/

# Verify data structure
tree -L 2 data/markerless_mouse_1_nerf/

# Expected structure:
# data/markerless_mouse_1_nerf/
# ├── videos_undist/
# │   ├── 0.mp4 ... 5.mp4
# ├── simpleclick_undist/
# │   ├── 0.mp4 ... 5.mp4
# ├── camera_params.h5
# └── keypoints2d_undist/

# Verify video files
ls -lh data/markerless_mouse_1_nerf/videos_undist/
# Should show: 6 MP4 files (~20MB each)

ls -lh data/markerless_mouse_1_nerf/simpleclick_undist/
# Should show: 6 MP4 files (~10MB each)

# Verify camera params
ls -lh data/markerless_mouse_1_nerf/camera_params.h5
# Should show: ~5KB file
```

**Checklist**:
- [ ] Data directory created
- [ ] 6 video files present (videos_undist/)
- [ ] 6 mask files present (simpleclick_undist/)
- [ ] Camera params file exists (camera_params.h5)
- [ ] File sizes are reasonable (~128MB total)

---

### Step 4: Environment Configuration ✅

```bash
# Add to ~/.bashrc for persistence
echo 'export PYTHONPATH="/home/YOUR_USERNAME/dev/pose-splatter:${PYTHONPATH}"' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc

# Apply changes
source ~/.bashrc

# Verify
echo $PYTHONPATH
echo $PYTORCH_CUDA_ALLOC_CONF
```

**Checklist**:
- [ ] PYTHONPATH includes project root
- [ ] PYTORCH_CUDA_ALLOC_CONF set to expandable_segments:True
- [ ] Changes persisted in ~/.bashrc

---

### Step 5: Preprocessing Pipeline ✅

```bash
# Activate environment
conda activate splatter

# Edit config paths (update YOUR_USERNAME)
nano configs/experiments/2d_gs_a6000.json
# Update:
# - "data_directory": "/home/YOUR_USERNAME/dev/pose-splatter/data/markerless_mouse_1_nerf/"
# - "project_directory": "/home/YOUR_USERNAME/dev/pose-splatter/output/2d_gs_a6000/"

# Run preprocessing (if not already done)
conda run -n splatter python scripts/preprocessing/estimate_up_direction.py \
    configs/experiments/2d_gs_a6000.json

conda run -n splatter python scripts/preprocessing/calculate_center_rotation.py \
    configs/experiments/2d_gs_a6000.json

conda run -n splatter python scripts/preprocessing/calculate_crop_indices.py \
    configs/experiments/2d_gs_a6000.json

conda run -n splatter python scripts/preprocessing/write_images.py \
    configs/experiments/2d_gs_a6000.json

conda run -n splatter python scripts/preprocessing/copy_to_zarr.py \
    output/2d_gs_a6000/images/images.h5 \
    output/2d_gs_a6000/images/images.zarr

# Verify outputs
ls -lh output/2d_gs_a6000/images/
# Should show: images.h5, images.zarr

ls -lh output/2d_gs_a6000/
# Should show: volumes/, images/, center_rotation.npz, volume_sum.npy
```

**Checklist**:
- [ ] Config file paths updated
- [ ] Up direction estimated
- [ ] Center rotation calculated
- [ ] Crop indices calculated
- [ ] Images written to HDF5
- [ ] Data copied to ZARR format
- [ ] All output files present

---

### Step 6: Debug Mode Training ✅

**CRITICAL: Always run debug mode before full training!**

```bash
# Debug mode (5-10 minutes)
conda run -n splatter python scripts/training/train_script.py \
    configs/debug/2d_3d_comparison_2d_debug.json \
    --epochs 1 --max_batches 10

# Check output
# Expected:
# - ✓ Using 2D Gaussian Splatting renderer
# - Loss decreasing (2.36 → 2.25 → ...)
# - No CUDA OOM errors
# - No gradient propagation errors

# Check logs
tail -100 output/logs/2d_debug_*.log
```

**Success Criteria**:
- [ ] Training started without errors
- [ ] GPU detected and used
- [ ] Loss values decreasing
- [ ] No CUDA OOM errors
- [ ] No gradient propagation errors
- [ ] Completed 10 batches successfully

**If Debug Fails**:
- Check error message in log
- Verify PYTHONPATH and conda environment
- Ensure GPU has enough free memory (`nvidia-smi`)
- Review troubleshooting section in README

---

### Step 7: Full Training Launch ✅

**Only proceed if debug mode succeeded!**

```bash
# Launch full training (4-8 hours)
nohup conda run -n splatter python scripts/training/train_script.py \
    configs/experiments/2d_gs_a6000.json \
    --epochs 50 \
    > output/logs/2d_gs_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Note the job ID
echo $!

# Monitor progress
tail -f output/logs/2d_gs_full_*.log

# Monitor GPU
watch -n 2 nvidia-smi
```

**Expected Performance**:
- Memory usage: 15-18GB
- Speed: ~8-12s per batch
- Total time: 4-8 hours (50 epochs)
- Loss: Should decrease steadily

**Monitoring Commands**:
```bash
# Check training progress
tail -f output/logs/2d_gs_full_*.log

# Check GPU utilization
nvidia-smi dmon -s u

# Check if process is running
ps aux | grep train_script.py

# Get job ID if needed
pgrep -f "train_script.py"
```

**Checklist**:
- [ ] Training launched successfully
- [ ] nohup prevents SSH disconnection
- [ ] Log file being written
- [ ] GPU utilization >70%
- [ ] Memory usage 15-18GB
- [ ] No OOM errors in first hour

---

### Step 8: Post-Training Evaluation ✅

```bash
# Wait for training to complete
# Check if process finished:
ps aux | grep train_script.py
# (should return nothing when done)

# Verify checkpoint exists
ls -lh output/2d_gs_a6000/checkpoint.pt
# Should show: ~100-200MB file

# Run evaluation
conda run -n splatter python scripts/training/evaluate_model.py \
    configs/experiments/2d_gs_a6000.json

# Check metrics
cat output/2d_gs_a6000/metrics_test.csv

# Expected good results:
# - PSNR > 25 dB
# - SSIM > 0.8
# - IoU > 0.7
# - L1 < 0.1
```

**Checklist**:
- [ ] Training completed without errors
- [ ] Checkpoint file created (~100-200MB)
- [ ] Evaluation ran successfully
- [ ] Metrics CSV generated
- [ ] PSNR > 25 dB
- [ ] Results look reasonable

---

### Step 9: Visualization ✅

```bash
# Generate all visualizations
bash scripts/visualization/run_all_visualization.sh

# View results
ls -lh output/2d_gs_a6000/renders/

# Expected outputs:
# - multiview/ (6 camera views)
# - temporal/ (30 frames + video)
# - rotation360/ (24 angles + video)

# Interactive visualization
conda run -n splatter python scripts/visualization/visualize_gaussian_rerun.py \
    output/2d_gs_a6000/gaussians/gaussian_frame0000.npz
```

**Checklist**:
- [ ] Visualizations generated
- [ ] Multi-view renders look correct
- [ ] Temporal sequence shows motion
- [ ] 360-degree rotation works
- [ ] Point clouds exported

---

## Comparison: 2D vs 3D GS

After 2D GS training completes, run comparison experiment:

```bash
# Run comparison (trains both 2D and 3D)
bash scripts/experiments/run_2d_3d_comparison.sh --phase1

# Analyze results
bash scripts/experiments/run_2d_3d_comparison.sh --phase2

# View comparison
cat output/2d_3d_comparison/comparison_report.txt
```

**Expected Findings**:
- 2D GS: Higher PSNR (~27-30 dB), slower training
- 3D GS: Good PSNR (~25-28 dB), faster, less memory

---

## Troubleshooting

### Issue 1: CUDA OOM (Out of Memory)

**Symptoms**:
```
torch.OutOfMemoryError: CUDA out of memory
```

**Solutions**:
1. Check for zombie processes:
   ```bash
   nvidia-smi | grep python
   pkill -9 -f "train_script.py"
   ```

2. Reduce config settings:
   ```json
   "image_downsample": 6,  // 4 → 6
   "gaussian_config": {
       "batch_size": 3  // 5 → 3
   }
   ```

3. Restart GPU:
   ```bash
   sudo nvidia-smi --gpu-reset
   ```

### Issue 2: Gradient Propagation Error

**Symptoms**:
```
RuntimeError: element 0 of tensors does not require grad
```

**Solutions**:
1. Verify latest code:
   ```bash
   git pull origin master
   grep "canvas = torch.zeros" src/gaussian_renderer.py
   # Should show: canvas = torch.zeros(...) + 0.0
   ```

2. Check renderer initialization:
   ```bash
   grep "gaussian_mode" configs/experiments/2d_gs_a6000.json
   # Should show: "gaussian_mode": "2d"
   ```

### Issue 3: Module Import Error

**Symptoms**:
```
ModuleNotFoundError: No module named 'src'
```

**Solutions**:
```bash
# Verify PYTHONPATH
echo $PYTHONPATH
# Should include: /home/YOUR_USERNAME/dev/pose-splatter

# Re-export if needed
export PYTHONPATH="/home/YOUR_USERNAME/dev/pose-splatter:${PYTHONPATH}"
```

### Issue 4: Slow Training (<50% GPU Utilization)

**Solutions**:
1. Increase batch size:
   ```json
   "gaussian_config": {
       "batch_size": 10  // 5 → 10
   }
   ```

2. Check data loading:
   ```bash
   # Verify num_workers in training script
   grep "num_workers" scripts/training/train_script.py
   ```

---

## Performance Benchmarks

### Expected Performance (A6000 24GB)

| Metric | Value |
|--------|-------|
| GPU Memory | 15-18GB |
| GPU Utilization | 70-90% |
| Training Speed | 8-12s/batch |
| Total Training Time | 4-8 hours (50 epochs) |
| Expected PSNR | 27-30 dB |
| Expected SSIM | 0.85-0.92 |

### Comparison: 2D vs 3D GS

| Feature | 2D GS | 3D GS (gsplat) |
|---------|-------|----------------|
| GPU Memory | 15-18GB | 4-6GB |
| Speed (per batch) | 8-12s | 16-20s |
| PSNR | 27-30 dB | 25-28 dB |
| Detail Quality | Higher | Good |
| GPU Requirement | 24GB+ | 12GB+ |

---

## Documentation References

- **2D GS Optimization**: `docs/reports/251116_2d_gaussian_optimization.md`
- **Project Structure**: `docs/reports/251115_project_reorganization.md`
- **Session Summary**: `docs/reports/251116_session_summary.md`
- **README**: `README.md` (See "A6000 GPU Setup Guide" section)

---

## Quick Reference Card

**Save this for quick access:**

```bash
# Setup (one-time)
conda activate splatter
export PYTHONPATH="/home/YOUR_USERNAME/dev/pose-splatter:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug test (5-10 min)
conda run -n splatter python scripts/training/train_script.py \
    configs/debug/2d_3d_comparison_2d_debug.json --epochs 1 --max_batches 10

# Full training (4-8 hours)
nohup conda run -n splatter python scripts/training/train_script.py \
    configs/experiments/2d_gs_a6000.json --epochs 50 \
    > output/logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor
tail -f output/logs/train_*.log
watch -n 2 nvidia-smi

# Evaluate
conda run -n splatter python scripts/training/evaluate_model.py \
    configs/experiments/2d_gs_a6000.json

# Emergency stop
pkill -9 -f "train_script.py"
```

---

## Final Checklist

### Pre-Training ✅
- [ ] GPU verified (A6000 24GB+)
- [ ] Environment setup complete
- [ ] Data downloaded and verified
- [ ] Config paths updated
- [ ] Preprocessing completed
- [ ] Debug mode successful

### During Training 🚀
- [ ] Training launched with nohup
- [ ] Log file being written
- [ ] GPU utilization >70%
- [ ] Memory usage 15-18GB
- [ ] No errors in first hour
- [ ] Loss decreasing steadily

### Post-Training ✅
- [ ] Training completed successfully
- [ ] Checkpoint file created
- [ ] Evaluation metrics good (PSNR >25)
- [ ] Visualizations generated
- [ ] Results documented

---

**Document Created**: 2025-11-16
**Last Updated**: 2025-11-16
**Author**: Claude Code (Anthropic)
**Version**: 1.0
**Target GPU**: NVIDIA A6000 (24GB+)
