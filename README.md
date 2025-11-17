## Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance [NeurIPS 2025]
[![arXiv](https://img.shields.io/badge/arXiv-2505.18342-b31b1b.svg)](https://arxiv.org/pdf/2505.18342.pdf) 

- Authors: [Jack Goffinet*](https://scholar.google.com/citations?user=-oXW2RYAAAAJ&hl=en),  [Youngjo Min*](https://sites.google.com/view/youngjo-min),  [Carlo Tomasi](https://users.cs.duke.edu/~tomasi/), [David Carlson](https://carlson.pratt.duke.edu/) (* denotes equal contribution)
<div align="center">

![Teaser Image](assets/teaser.png)

</div>


Code for "Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance."


### Quick Start

#### Prerequisites

**System Requirements**:
- **3D Gaussian Splatting (gsplat)**: 12GB+ GPU (RTX 3060, RTX 3090, etc.)
- **2D Gaussian Splatting (custom)**: 24GB+ GPU (A6000, RTX 4090, A100)
- Python 3.10+
- CUDA 11.8 or 12.8+

**Installation (conda environment)**:
```bash
# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gsplat torchmetrics matplotlib seaborn pandas tabulate h5py zarr
```

#### Dataset Setup

This project uses the **DANNCE `markerless_mouse_1` dataset** preprocessed by the MAMMAL_mouse project.

**Quick Start: Copy from MAMMAL_mouse Repository**

If you have already cloned the MAMMAL_mouse repository:

```bash
# Create target directory
mkdir -p data/markerless_mouse_1_nerf

# Copy preprocessed data from MAMMAL_mouse
cp -r /path/to/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \
      data/markerless_mouse_1_nerf/

# Example (if MAMMAL_mouse is in ~/dev):
cp -r ~/dev/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \
      data/markerless_mouse_1_nerf/
```

**Alternative: Download MAMMAL_mouse Repository**

```bash
# Clone MAMMAL_mouse repository
git clone https://github.com/kafkapple/MAMMAL_mouse.git /tmp/MAMMAL_mouse

# Copy example data
cp -r /tmp/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \
      data/markerless_mouse_1_nerf/
```

**Expected Data Structure**:
```
data/markerless_mouse_1_nerf/
├── videos_undist/              # 6-camera RGB videos (128MB)
│   ├── 0.mp4                  # Camera 0: 25MB
│   ├── 1.mp4 ... 5.mp4        # Cameras 1-5: 17-24MB each
├── simpleclick_undist/         # Segmentation masks (64MB)
│   ├── 0.mp4 ... 5.mp4        # Binary masks: 11MB each
├── keypoints2d_undist/         # 2D keypoint detections (55MB)
│   ├── result_view_0.pkl      # Keypoints for camera 0: 9.1MB
│   ├── result_view_1.pkl ... result_view_5.pkl
├── new_cam.pkl                 # Camera calibration (54MB)
├── add_labels_3d_8keypoints.pkl  # 3D keypoint labels (13KB)
├── label_ids.pkl               # Label ID mapping
└── readme.md                   # Dataset notes
```

**File Format Details**:
- **Videos (MP4)**: RGB videos and binary segmentation masks
- **Keypoints (PKL)**: Pickle files with 2D keypoint detections per camera
- **Camera Params (PKL)**: Camera intrinsics, extrinsics, distortion coefficients
- **Labels (PKL)**: 3D ground truth keypoints and label mappings

**Dataset Specifications**:
- **Cameras**: 6 synchronized views
- **Resolution**: ~1152 × 1024 pixels (varies by camera)
- **Frame Rate**: 100 FPS
- **Total Frames**: ~18,000 frames (note: jumps at frames 5900, 11800, 17700)
- **Subject**: Mouse in natural behavior

**Dataset Notes**:
- This dataset is preprocessed from the original DANNCE `markerless_mouse_1` dataset
- Videos have been undistorted and synchronized
- 2D keypoints have been detected using pose estimation models
- Binary masks have been generated using SimpleClick segmentation

**Original Data Sources**:
- Original DANNCE: [spoonsso/DANNCE](https://github.com/spoonsso/DANNCE)
- PyTorch Implementation: [tqxli/dannce-pytorch](https://github.com/tqxli/dannce-pytorch)
- MAMMAL_mouse: Preprocessed version with multi-animal tracking support

---

#### Automated Pipeline (Recommended)
Run the full pipeline with a single command:
```bash
bash run_pipeline_auto.sh
# or with custom config:
bash run_pipeline_auto.sh configs/your_config.json
```

This will automatically execute all steps from center calculation to model evaluation.

#### Monitor Progress
```bash
# Real-time monitoring (updates every 10 seconds)
watch -n 10 ./monitor_pipeline.sh

# Check GPU usage
watch -n 2 nvidia-smi

# View specific step log
tail -f output/markerless_mouse_nerf/logs/step6_training.log
```

---

### 🚀 A6000 GPU Setup Guide (24GB+ for 2D Gaussian Splatting)

This section provides a complete workflow for deploying this project on a high-memory GPU environment (A6000, RTX 4090, A100) specifically for **2D Gaussian Splatting** experiments.

#### Step 1: Clone Repository

```bash
# Clone the repository
cd ~/dev  # or your preferred directory
git clone https://github.com/YOUR_USERNAME/pose-splatter.git
cd pose-splatter
```

#### Step 2: Environment Setup

**Option 1: Create New Conda Environment (Recommended)**

```bash
# Create conda environment with Python 3.10
conda create -n splatter python=3.10 -y
conda activate splatter

# Install PyTorch with CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional dependencies
pip install gsplat torchmetrics matplotlib seaborn pandas tabulate h5py zarr einops

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Option 2: Use Existing Environment**

If you already have a compatible conda environment:

```bash
conda activate your_env
pip install -r requirements.txt  # Install missing packages if any
```

**Environment Verification**:

```bash
# Check GPU memory
nvidia-smi

# Expected output for A6000:
# GPU Name: NVIDIA RTX A6000
# Total Memory: 48GB (or 24GB for single A6000)

# Verify Python environment
which python  # Should point to conda environment
python --version  # Should be 3.10+
```

#### Step 3: Data Preparation

**Option 1: Copy from MAMMAL_mouse Repository (Recommended)**

If you have the MAMMAL_mouse repository cloned locally:

```bash
# Create data directory
mkdir -p data/markerless_mouse_1_nerf

# Copy preprocessed data
cp -r ~/dev/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \
      data/markerless_mouse_1_nerf/

# Or specify your MAMMAL_mouse path
cp -r /path/to/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \
      data/markerless_mouse_1_nerf/
```

**Option 2: Clone MAMMAL_mouse and Copy**

```bash
# Clone MAMMAL_mouse repository
git clone https://github.com/kafkapple/MAMMAL_mouse.git /tmp/MAMMAL_mouse

# Copy example data
cp -r /tmp/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \
      data/markerless_mouse_1_nerf/
```

**Option 3: Transfer from Remote Server**

```bash
# If dataset is on another machine
rsync -avz --progress user@server:/path/to/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/ \
      data/markerless_mouse_1_nerf/
```

**Expected Data Structure**:

```bash
data/markerless_mouse_1_nerf/
├── videos_undist/              # 6-camera RGB videos (128MB)
│   ├── 0.mp4                  # Camera 0: 25MB
│   ├── 1.mp4 ... 5.mp4        # Cameras 1-5: 17-24MB each
├── simpleclick_undist/         # Segmentation masks (64MB)
│   ├── 0.mp4 ... 5.mp4        # Binary masks: 11MB each
├── keypoints2d_undist/         # 2D keypoint detections (55MB)
│   ├── result_view_0.pkl      # Keypoints for camera 0: 9.1MB
│   ├── result_view_1.pkl ... result_view_5.pkl
├── new_cam.pkl                 # Camera calibration (54MB)
├── add_labels_3d_8keypoints.pkl  # 3D keypoint labels (13KB)
├── label_ids.pkl               # Label ID mapping
└── readme.md                   # Dataset notes
```

**Verify Dataset**:

```bash
# Quick verification - check file counts and sizes
ls -lh data/markerless_mouse_1_nerf/videos_undist/      # Should show 6 MP4 files
ls -lh data/markerless_mouse_1_nerf/simpleclick_undist/ # Should show 6 MP4 files
ls -lh data/markerless_mouse_1_nerf/keypoints2d_undist/ # Should show 6 PKL files
du -sh data/markerless_mouse_1_nerf/                    # Should show ~299MB

# Comprehensive verification (recommended)
python scripts/utils/verify_mammal_data.py

# This checks:
# - All 6 camera RGB videos (videos_undist/)
# - All 6 segmentation masks (simpleclick_undist/)
# - All 6 keypoint files (keypoints2d_undist/)
# - Camera calibration (new_cam.pkl)
# - 3D keypoint labels (add_labels_3d_8keypoints.pkl)
# - Label ID mapping (label_ids.pkl)
```

#### Step 4: Preprocessing Pipeline

Run preprocessing steps to prepare data for training:

```bash
# Set environment variables (add to ~/.bashrc for persistence)
export PYTHONPATH="$(pwd):${PYTHONPATH}"  # Or use absolute path
# export PYTHONPATH="/home/YOUR_USERNAME/dev/pose-splatter:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run full preprocessing pipeline
bash scripts/preprocessing/run_full_preprocessing.sh configs/baseline/markerless_mouse_nerf.json

# Or run steps individually:
# Step 1: Estimate up direction
conda run -n splatter python scripts/preprocessing/estimate_up_direction.py configs/baseline/markerless_mouse_nerf.json

# Step 2: Calculate center and rotation
conda run -n splatter python scripts/preprocessing/calculate_center_rotation.py configs/baseline/markerless_mouse_nerf.json

# Step 3: Calculate crop indices
conda run -n splatter python scripts/preprocessing/calculate_crop_indices.py configs/baseline/markerless_mouse_nerf.json

# Step 4: Write images to HDF5
conda run -n splatter python scripts/preprocessing/write_images.py configs/baseline/markerless_mouse_nerf.json

# Step 5: Copy to ZARR for training
conda run -n splatter python scripts/preprocessing/copy_to_zarr.py \
    output/markerless_mouse_nerf/images/images.h5 \
    output/markerless_mouse_nerf/images/images.zarr
```

**Monitor Preprocessing**:

```bash
# Check logs
tail -f output/markerless_mouse_nerf/logs/preprocessing.log

# Verify outputs
ls -lh output/markerless_mouse_nerf/images/
ls -lh output/markerless_mouse_nerf/volumes/
```

#### Step 5: Training with 2D Gaussian Splatting

**2D GS Configuration**:

For A6000 24GB GPU, use the following configuration:

```json
// configs/experiments/2d_gs_a6000.json
{
    "data_directory": "/home/YOUR_USERNAME/dev/pose-splatter/data/markerless_mouse_1_nerf/",
    "project_directory": "/home/YOUR_USERNAME/dev/pose-splatter/output/2d_gs_full/",
    "image_downsample": 4,      // 256×288 images
    "grid_size": 112,            // UNet compatible
    "max_frames": 50,            // Full dataset
    "gaussian_mode": "2d",       // Use 2D GS renderer
    "gaussian_config": {
        "sigma_cutoff": 3.0,
        "kernel_size": 5,
        "batch_size": 5          // Adjust based on GPU memory
    },
    "lr": 1e-4,
    "img_lambda": 0.5,
    "ssim_lambda": 0.0,
    "valid_every": 5,
    "plot_every": 1,
    "save_every": 5
}
```

**Run 2D GS Training**:

```bash
# Debug mode first (5-10 minutes, verify everything works)
conda run -n splatter python scripts/training/train_script.py \
    configs/debug/2d_3d_comparison_2d_debug.json \
    --epochs 1 --max_batches 10

# Check debug results
tail -100 output/logs/2d_debug_*.log

# If debug succeeds, run full training (4-8 hours)
nohup conda run -n splatter python scripts/training/train_script.py \
    configs/experiments/2d_gs_a6000.json \
    --epochs 50 \
    > output/logs/2d_gs_full_training.log 2>&1 &

# Monitor training progress
tail -f output/logs/2d_gs_full_training.log

# Monitor GPU usage
watch -n 2 nvidia-smi
```

**Expected Performance (A6000 24GB)**:

| Metric | Value |
|--------|-------|
| Image Size | 256×288 (downsample=4) |
| Batch Size | 5-10 Gaussians |
| Memory Usage | ~15-18GB |
| Speed | ~8-12s per training batch |
| Training Time | 4-8 hours (50 epochs) |
| Expected PSNR | >25 dB |

**Memory Optimization Tips**:

If you encounter OOM errors even on A6000:

```bash
# Option 1: Reduce image size
"image_downsample": 6,  # 4 → 6 (192×216 images)

# Option 2: Reduce batch size
"batch_size": 3,  # 5 → 3

# Option 3: Reduce max frames
"max_frames": 30,  # 50 → 30

# Option 4: Use gradient checkpointing (if implemented)
"use_gradient_checkpointing": true
```

#### Step 6: Evaluation and Visualization

```bash
# Evaluate trained model
conda run -n splatter python scripts/training/evaluate_model.py \
    configs/experiments/2d_gs_a6000.json

# Generate visualizations
bash scripts/visualization/run_all_visualization.sh

# View results
ls -lh output/2d_gs_full/renders/
ls -lh output/2d_gs_full/metrics_test.csv
```

#### Step 7: Compare 2D vs 3D Gaussian Splatting

**Run Comparison Experiment**:

```bash
# Phase 1: Train both 2D and 3D (10 epochs each)
bash scripts/experiments/run_2d_3d_comparison.sh --phase1

# Monitor progress
tail -f output/logs/2d_debug_*.log
tail -f output/logs/3d_debug_*.log

# Phase 2: Analyze results
bash scripts/experiments/run_2d_3d_comparison.sh --phase2

# View comparison report
cat output/2d_3d_comparison/comparison_report.txt
```

**Expected Comparison Results**:

| Feature | 2D GS | 3D GS (gsplat) |
|---------|-------|----------------|
| GPU Memory | 15-18GB | 4-6GB |
| Training Speed | 8-12s/batch | 16-20s/batch |
| Image Quality (PSNR) | ~27-30 dB | ~25-28 dB |
| Reconstruction Detail | Higher | Good |
| GPU Requirement | 24GB+ | 12GB+ |
| Use Case | Research, high quality | Production, efficiency |

#### Troubleshooting for A6000 Environment

**Common Issues**:

1. **CUDA OOM despite 24GB GPU**:
   ```bash
   # Check for zombie processes
   nvidia-smi | grep python
   pkill -9 -f "train_script.py"

   # Reduce settings
   # Edit config: image_downsample: 6, batch_size: 3
   ```

2. **Gradient propagation errors**:
   ```bash
   # Verify you have the latest gaussian_renderer.py
   grep "canvas = torch.zeros" src/gaussian_renderer.py
   # Should show: canvas = torch.zeros(...) + 0.0  # Non-leaf tensor
   ```

3. **Module import errors**:
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH="$(pwd):${PYTHONPATH}"  # Or use absolute path
# export PYTHONPATH="/home/YOUR_USERNAME/dev/pose-splatter:${PYTHONPATH}"

   # Verify
   echo $PYTHONPATH
   ```

4. **Slow training (GPU underutilized)**:
   ```bash
   # Check GPU utilization
   nvidia-smi dmon -s u

   # If <50%, increase batch_size
   "batch_size": 10  # or higher
   ```

**Performance Benchmarks**:

```bash
# Run benchmark script
conda run -n splatter python scripts/utils/benchmark_renderer.py \
    --mode 2d --batch_size 5 --image_size 256 288

# Compare 2D vs 3D
conda run -n splatter python scripts/utils/benchmark_renderer.py \
    --compare --batch_sizes 1 5 10
```

#### Best Practices for A6000 Deployment

1. **Always run debug mode first** (saves hours if config is wrong)
2. **Monitor GPU memory** with `watch -n 2 nvidia-smi`
3. **Use `nohup` for long training** to prevent SSH disconnection
4. **Save checkpoints frequently** (`save_every: 5`)
5. **Log everything** (redirect stdout/stderr to log files)
6. **Document hyperparameters** in git commit messages

#### Quick Reference Commands

```bash
# Setup (one-time)
conda activate splatter
export PYTHONPATH="$(pwd):${PYTHONPATH}"  # Or use absolute path
# export PYTHONPATH="/home/YOUR_USERNAME/dev/pose-splatter:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug test (5-10 minutes)
conda run -n splatter python scripts/training/train_script.py \
    configs/debug/2d_3d_comparison_2d_debug.json --epochs 1 --max_batches 10

# Full training (4-8 hours)
nohup conda run -n splatter python scripts/training/train_script.py \
    configs/experiments/2d_gs_a6000.json --epochs 50 \
    > output/logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor
tail -f output/logs/training_*.log
watch -n 2 nvidia-smi

# Evaluate
conda run -n splatter python scripts/training/evaluate_model.py \
    configs/experiments/2d_gs_a6000.json
```

**Documentation References**:
- 2D GS Optimization Guide: `docs/reports/251116_2d_gaussian_optimization.md`
- Project Structure: `docs/reports/251115_project_reorganization.md`
- Troubleshooting: `docs/reports/251116_session_summary.md`

---

### Manual Usage Steps

1. **Collect up direction:**
    ```bash
    python3 estimate_up_direction.py configs/markerless_mouse_nerf.json
    ```

2. **Get center and rotations:**
    ```bash
    python3 calculate_center_rotation.py configs/markerless_mouse_nerf.json
    ```

3. **Get volume sum, decide volume_idx:**
    ```bash
    python3 calculate_crop_indices.py configs/markerless_mouse_nerf.json
    ```

4. **Write images in HDF5 format:**
    ```bash
    python3 write_images.py configs/markerless_mouse_nerf.json
    ```

5. **Copy images to ZARR for training:**
    ```bash
    python3 copy_to_zarr.py \
        output/markerless_mouse_nerf/images/images.h5 \
        output/markerless_mouse_nerf/images/images.zarr
    ```

6. **Train a model:**
    ```bash
    python3 train_script.py configs/markerless_mouse_nerf.json --epochs 50
    ```

7. **Evaluate model:**
    ```bash
    python3 evaluate_model.py configs/markerless_mouse_nerf.json
    ```

8. **Render an image:**
    ```bash
    python3 render_image.py configs/markerless_mouse_nerf.json <frame_idx> <view_idx>
    ```

9. **Calculate visual features:**
    ```bash
    python3 calculate_visual_features.py configs/markerless_mouse_nerf.json
    ```

10. **Calculate visual embedding:**
    ```bash
    python3 calculate_visual_embedding.py configs/markerless_mouse_nerf.json
    ```

---

### Analysis & Visualization

After training completes, analyze your results:

```bash
# Comprehensive analysis with metrics and plots
python3 analyze_results.py configs/markerless_mouse_nerf.json

# Visualize training curves
python3 visualize_training.py \
    --log_file output/markerless_mouse_nerf/logs/step6_training.log \
    --output_dir output/markerless_mouse_nerf/analysis

# Compare ground truth vs predictions
python3 visualize_renders.py \
    --mode compare \
    --gt_file output/markerless_mouse_nerf/images/images.h5 \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 500 1000 \
    --output_dir output/markerless_mouse_nerf/visualization
```

For detailed analysis guide, see [docs/reports/ANALYSIS_GUIDE.md](docs/reports/ANALYSIS_GUIDE.md)

---

### Rendering & Result Comparison

#### Novel View Synthesis

Pose Splatter performs **novel view synthesis** - generating new views of the animal from different camera angles using the trained 3D Gaussian model. This is not image upsampling, but rather creating completely new viewpoints.

#### Comprehensive Visualization Pipeline (NEW!)

After training, generate all visualization types with a single command:

```bash
# Run complete visualization pipeline
bash run_all_visualization.sh

# This generates:
# 1. Multi-view renders (6 camera angles for frame 0)
# 2. Temporal sequence (30 consecutive frames + video)
# 3. 360-degree rotation (24 angles + video)
# 4. 3D point cloud export (PLY format)
```

**Individual Visualization Scripts**:

```bash
# 1. Multi-view rendering (all 6 cameras)
python3 generate_multiview.py

# 2. Temporal sequence with video
python3 generate_temporal_video.py

# 3. 360-degree rotation views
python3 generate_360_rotation.py

# 4. Export 3D point cloud
python3 export_point_cloud.py --frame 0 --output pointcloud.ply
```

**Output Structure**:
```
output/markerless_mouse_nerf/
├── renders/
│   ├── multiview/          # 6 camera views of frame 0
│   │   ├── frame0000_view0.png
│   │   └── ... view1-5.png
│   ├── temporal/           # Temporal sequence
│   │   ├── frame0000.png ... frame0029.png
│   │   └── temporal_sequence.mp4
│   └── rotation360/        # 360-degree rotation
│       ├── rot000.png ... rot023.png
│       └── rotation360.mp4
└── pointclouds/
    └── frame0000.ply       # 3D Gaussian point cloud
```

**Prerequisites for Video Generation**:
```bash
# Install FFmpeg with x264 codec
conda install -c conda-forge x264 ffmpeg
```

**Point Cloud Viewing**:
- Use MeshLab, CloudCompare, or Blender to view `.ply` files
- Contains 3D positions, RGB colors, and opacity values

For detailed implementation and troubleshooting, see [VISUALIZATION_REPORT.md](VISUALIZATION_REPORT.md)

---

### 3D Export & Blender Integration (NEW!)

Export your trained 3D Gaussian Splatting model for use in external 3D software.

#### Single Frame Export

**Export Complete Gaussian Parameters**:
```bash
# Export as NumPy NPZ (recommended for research)
python3 export_gaussian_full.py --frame 0 --format npz

# Export as extended PLY (for Gaussian Splatting viewers)
python3 export_gaussian_full.py --frame 0 --format ply_extended

# Export as JSON (for inspection)
python3 export_gaussian_full.py --frame 0 --format json
```

**Export Basic Point Cloud**:
```bash
# Export single frame as PLY
python3 export_point_cloud.py --frame 0 --output pointcloud.ply
```

#### Multi-Frame Animation Sequence

Export multiple frames for animation workflows:

```bash
# Export 30 frames (PLY + NPZ)
python3 export_animation_sequence.py \
    --start_frame 0 \
    --num_frames 30 \
    --ply --npz

# Export 60 frames with JSON metadata
python3 export_animation_sequence.py \
    --start_frame 0 \
    --num_frames 60 \
    --ply --npz --json
```

**Output Structure**:
```
output/markerless_mouse_nerf/animation/
├── pointclouds/            # PLY point clouds
│   ├── frame0000.ply       # 16,000 points each (~1.2MB)
│   ├── frame0001.ply
│   └── ...
├── gaussians_npz/          # Full Gaussian parameters
│   ├── frame0000.npz       # Complete params (~730KB)
│   ├── frame0001.npz
│   └── ...
└── gaussians_json/         # Metadata (optional)
    ├── frame0000.json
    └── ...
```

#### Blender Integration

Import your 3D reconstructions into Blender:

```bash
# Option 1: Use Blender GUI
# 1. Open Blender
# 2. Switch to Scripting workspace
# 3. Open: blender_import_pointcloud.py
# 4. Edit the file paths at the bottom
# 5. Run script (Alt+P)

# Option 2: Command line
blender --background --python blender_import_pointcloud.py
```

**Blender Script Features**:
- Import single PLY file as particle system
- Import NPZ as Gaussian instances with rotation/scale
- Import animation sequence with keyframe visibility
- Automatic material creation with vertex colors

**Edit the script to choose import mode**:
```python
# In blender_import_pointcloud.py, uncomment your preferred option:

# Single PLY file
PLY_FILE = "output/markerless_mouse_nerf/pointclouds/frame0000.ply"

# NPZ Gaussian data
# NPZ_FILE = "output/markerless_mouse_nerf/gaussians/gaussian_frame0000.npz"

# Animation sequence
# PLY_DIR = "output/markerless_mouse_nerf/animation/pointclouds"
```

#### Export Formats Comparison

| Format | Size | Use Case | Contains |
|--------|------|----------|----------|
| **PLY** | 1.2MB | Viewing in 3D software | Position, RGB, opacity |
| **NPZ** | 730KB | Research & analysis | All Gaussian params (means, quats, scales, opacities, colors) |
| **JSON** | 54KB | Inspection & debugging | Metadata + 100 sample Gaussians |
| **PLY Extended** | ~2MB | Advanced Gaussian viewers | PLY + quaternions + scales |

#### Supported Software & Visualization Tools

- **Rerun.io** ⭐: Interactive 3D visualization with timeline (NPZ, animation sequences)
- **MeshLab**: PLY viewing and editing
- **CloudCompare**: Point cloud analysis
- **Blender**: Animation and rendering (PLY, NPZ)
- **Python (matplotlib)**: Static visualization and analysis (NPZ)
- **Gaussian Splatting Viewers**: Extended PLY format

#### Example Workflow: Blender Animation

```bash
# 1. Export 60-frame sequence
python3 export_animation_sequence.py --start_frame 0 --num_frames 60 --ply

# 2. Create temporal video (optional)
python3 generate_temporal_video.py

# 3. Import into Blender
blender --background --python blender_import_pointcloud.py

# 4. In Blender:
#    - Set frame range 1-60
#    - Add camera animation
#    - Set up lighting
#    - Render animation
```

**Tips for Blender**:
- Enable GPU rendering for faster previews
- Use EEVEE for real-time preview
- Use Cycles for final quality renders
- Adjust particle size in Point Cloud settings (0.001-0.01)
- Enable vertex color display in Material Preview mode

---

#### Interactive Visualization with Rerun.io ⭐ NEW!

Real-time 3D interactive visualization powered by [Rerun.io](https://rerun.io/).

**Installation**:
```bash
pip install rerun-sdk
```

**Single Frame Visualization**:
```bash
# Launch interactive 3D viewer (requires display)
python3 visualize_gaussian_rerun.py output/markerless_mouse_nerf/gaussians/gaussian_frame0000.npz

# Custom point size
python3 visualize_gaussian_rerun.py file.npz --point_size 0.01

# SSH/Headless environments - Save to .rrd file
python3 visualize_gaussian_rerun.py file.npz --save output.rrd
# Then download and view locally: rerun output.rrd

# SSH/Headless environments - Web server mode
python3 visualize_gaussian_rerun.py file.npz --web 9090
# Open http://localhost:9090 in browser (or forward port via SSH)
```

**Animation Sequence with Timeline**:
```bash
# Interactive timeline for animation (requires display)
python3 visualize_gaussian_rerun.py output/markerless_mouse_nerf/animation/gaussians_npz/ --sequence

# SSH/Headless - Save animation to .rrd file
python3 visualize_gaussian_rerun.py output/markerless_mouse_nerf/animation/gaussians_npz/ --sequence --save animation.rrd

# SSH/Headless - Web server for animation
python3 visualize_gaussian_rerun.py output/markerless_mouse_nerf/animation/gaussians_npz/ --sequence --web 9090
```

**Compare Multiple Frames Side-by-Side**:
```bash
python3 visualize_gaussian_rerun.py \
    --compare \
    output/markerless_mouse_nerf/animation/gaussians_npz/frame0000.npz \
    output/markerless_mouse_nerf/animation/gaussians_npz/frame0015.npz \
    output/markerless_mouse_nerf/animation/gaussians_npz/frame0029.npz
```

**Rerun Features**:
- **Interactive 3D Navigation**: Rotate, pan, zoom with mouse
- **Layer Toggle**: Show/hide different visualizations (RGB, opacity, scale)
- **Timeline Control**: Scrub through animation frames
- **Real-time Updates**: Instant visualization without preprocessing
- **Multi-view**: Compare multiple frames simultaneously
- **Statistics Display**: Live Gaussian parameter statistics

**SSH/Headless Environment Support**:

If you encounter display errors (e.g., `WAYLAND_DISPLAY not set`), use one of these alternatives:

**Option 1: Save to .rrd file (Recommended for SSH)**:
```bash
# Save visualization to file
python3 visualize_gaussian_rerun.py file.npz --save output.rrd

# Download the .rrd file to your local machine, then view:
rerun output.rrd
```

**Option 2: Web server mode**:
```bash
# Terminal 1: Start Rerun web server
rerun --port 9090

# Terminal 2: Connect visualization to the server
python3 visualize_gaussian_rerun.py file.npz --web 9090

# For SSH access, forward the port (on your local machine):
ssh -L 9090:localhost:9090 user@server

# Open browser: http://localhost:9090
```

Both options work with `--sequence` and `--compare` modes.

**권장 방법 (SSH 환경)**:
- `.rrd` 파일로 저장 → 로컬 컴퓨터로 다운로드 후 `rerun` 명령으로 보기
- 728KB 파일 크기로 다운로드가 빠르고 간편합니다

**Visualization Layers**:
- `gaussian/points`: RGB colored point cloud
- `gaussian/opacity`: Grayscale visualization by opacity
- `gaussian/scale`: Color-coded by Gaussian scale
- `stats`: Text overlay with parameter statistics

**Alternative: Matplotlib Static Visualization**:
```bash
# Generate static plots with matplotlib
python3 visualize_gaussian.py output/markerless_mouse_nerf/gaussians/gaussian_frame0000.npz

# Output: gaussian_frame0000_visualization.png (6 subplots)
# - 3D point cloud (RGB, opacity, scale)
# - Histograms (opacity, scale, color distributions)
```

**Example Usage Scenarios**:

*Quick Preview - Single Frame*:
```bash
# View the first reconstructed frame
python3 visualize_gaussian_rerun.py \
    exports/markerless_mouse_nerf_20251110_150641/gaussians/gaussian_frame0000.npz
```

*Analyze Motion - Animation Timeline*:
```bash
# Scrub through 30-frame animation sequence
python3 visualize_gaussian_rerun.py \
    exports/markerless_mouse_nerf_20251110_150641/animation/gaussians_npz/ \
    --sequence
```

*Compare Frames - Side-by-Side View*:
```bash
# Compare start, middle, and end frames
python3 visualize_gaussian_rerun.py --compare \
    exports/markerless_mouse_nerf_20251110_150641/animation/gaussians_npz/frame0000.npz \
    exports/markerless_mouse_nerf_20251110_150641/animation/gaussians_npz/frame0014.npz \
    exports/markerless_mouse_nerf_20251110_150641/animation/gaussians_npz/frame0029.npz
```

*Adjust Visualization Quality*:
```bash
# Larger points for better visibility
python3 visualize_gaussian_rerun.py file.npz --point_size 0.01

# Smaller points for dense clouds
python3 visualize_gaussian_rerun.py file.npz --point_size 0.002
```

**Rerun Viewer Controls**:
- **Mouse Navigation**:
  - Left click + drag: Rotate camera
  - Right click + drag: Pan camera
  - Scroll wheel: Zoom in/out
  - Middle click + drag: Pan camera
- **Layer Panel** (left sidebar):
  - Toggle `gaussian/points` for RGB visualization
  - Toggle `gaussian/opacity` for opacity heatmap
  - Toggle `gaussian/scale` for scale heatmap
  - Toggle `stats` for parameter statistics
- **Timeline** (bottom, sequence mode only):
  - Drag slider to scrub frames
  - Click ▶ to play animation
  - Adjust playback speed

**What You'll See**:
- **RGB View**: 16,000 colored 3D points representing the mouse
- **Opacity View**: Grayscale visualization showing transparency values
- **Scale View**: Color-coded by Gaussian size (red = large, blue = small)
- **Statistics Panel**: Real-time parameter stats (mean, std dev)

---

#### Render Specific Frames/Views

After training completes, you can render specific frames and camera views:

```bash
# Render a specific frame and view
# Usage: render_image.py <config> <frame_idx> <view_idx>
python3 render_image.py configs/markerless_mouse_nerf.json 100 0

# Examples:
python3 render_image.py configs/markerless_mouse_nerf.json 0 0      # First frame, first camera
python3 render_image.py configs/markerless_mouse_nerf.json 500 2    # Frame 500, camera 2
python3 render_image.py configs/markerless_mouse_nerf.json 1000 5   # Frame 1000, camera 5

# Note: frame_idx range is 0 to (total_frames - 1)
# Note: view_idx range is 0 to (num_cameras - 1), typically 0-5 for 6 cameras
```

**Output**: Single rendered image saved in the output directory

#### Automatic Evaluation & Comparison

The `evaluate_model.py` script automatically renders all test frames and computes metrics:

```bash
# Full evaluation with metrics
python3 evaluate_model.py configs/markerless_mouse_nerf.json

# This will:
# 1. Load the trained model checkpoint
# 2. Render all test set frames
# 3. Compare with ground truth
# 4. Calculate metrics: PSNR, SSIM, IoU, L1 loss
# 5. Save results to:
#    - output/markerless_mouse_nerf/metrics_test.csv
#    - output/markerless_mouse_nerf/images/rendered_images.h5
```

**Output Metrics**:
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (>25 dB is good)
- **SSIM** (Structural Similarity): Higher is better (>0.8 is good)
- **IoU** (Intersection over Union): Higher is better (>0.7 is good)
- **L1 Loss**: Lower is better (<0.1 is good)

#### Visual Comparison

Use `visualize_renders.py` for side-by-side visual comparison:

```bash
# Compare ground truth vs predictions for specific frames
python3 visualize_renders.py \
    --mode compare \
    --gt_file output/markerless_mouse_nerf/images/images.h5 \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 100 500 1000 1500 \
    --output_dir output/markerless_mouse_nerf/visualization

# View only predictions in grid format
python3 visualize_renders.py \
    --mode grid \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 100 200 300 400 \
    --output_dir output/markerless_mouse_nerf/visualization
```

**Output**:
- `comparison_frame_XXXXX.png`: Side-by-side GT vs prediction with alpha channel
- `pred_grids/frame_XXXXX.png`: Multi-view grid of predicted renders
- `gt_grids/frame_XXXXX.png`: Multi-view grid of ground truth

#### Complete Workflow Example

```bash
# 1. Train the model (50 epochs, ~8-12 hours)
python3 train_script.py configs/markerless_mouse_nerf.json --epochs 50

# 2. Evaluate on test set (automatic rendering + metrics)
python3 evaluate_model.py configs/markerless_mouse_nerf.json

# 3. Analyze quantitative results
python3 analyze_results.py configs/markerless_mouse_nerf.json

# 4. Create visual comparisons
python3 visualize_renders.py \
    --mode compare \
    --gt_file output/markerless_mouse_nerf/images/images.h5 \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 500 1000 1500 \
    --output_dir output/markerless_mouse_nerf/visualization

# 5. Visualize training progress
python3 visualize_training.py \
    --log_file output/markerless_mouse_nerf/logs/step6_training.log \
    --output_dir output/markerless_mouse_nerf/analysis

# 6. (Optional) Render specific interesting frames
python3 render_image.py configs/markerless_mouse_nerf.json 1234 3
```

#### Understanding the Results

**Good Results Indicators**:
- PSNR > 25 dB: High image quality
- SSIM > 0.8: Good structural similarity
- IoU > 0.7: Accurate segmentation mask
- L1 < 0.1: Low pixel-level error
- Visual inspection: Sharp details, correct pose, minimal artifacts

**Common Issues**:
- Blurry renders → Increase grid_size or reduce image_downsample
- Missing details → Train longer or reduce learning rate
- Poor specific views → Check camera calibration
- Low IoU → Adjust volume_fill_color or shape carving parameters

---

### Experiment Configurations

Multiple experiment configurations are available in `configs/`:

- **baseline**: `markerless_mouse_nerf.json` - Standard configuration
- **high_res**: `markerless_mouse_nerf_high_res.json` - Higher resolution (2x downsample, grid 128)
- **fast**: `markerless_mouse_nerf_fast.json` - Fast prototyping (8x downsample, grid 64)
- **ssim**: `markerless_mouse_nerf_ssim.json` - With SSIM loss component

Compare configurations:
```bash
python3 compare_configs.py \
    configs/markerless_mouse_nerf.json \
    configs/markerless_mouse_nerf_high_res.json \
    --format markdown
```

---

### Documentation

- [Experiment Report](docs/reports/251109_experiment_baseline.md) - Current baseline experiment details
- [Analysis Guide](docs/reports/ANALYSIS_GUIDE.md) - Complete analysis workflow
- [Tools Summary](docs/reports/TOOLS_SUMMARY.md) - All available scripts and utilities

#### Visualization Reports (NEW!)
- **[Visualization Implementation Report](reports/VISUALIZATION_REPORT.md)** - Technical implementation details
- **[Safe Execution Guide](reports/SAFE_EXECUTION_GUIDE.md)** - GPU memory management guide
- **[Work Summary](reports/WORK_SUMMARY.md)** - Complete work summary
- **[Changelog](reports/CHANGELOG.md)** - Detailed change history

### Recent Updates (2025-11-10)

- ✅ **3D Export & Blender Integration**: Complete 3D data export system
  - Full Gaussian parameters export (NPZ, PLY Extended, JSON formats)
  - Multi-frame animation sequence export
  - Blender import scripts with particle system and keyframe animation
  - Support for MeshLab, CloudCompare, and custom viewers
- ✅ **Visualization Pipeline**: Complete implementation of 4 visualization types
  - Multi-view rendering (6 cameras)
  - Temporal sequence with video generation (60 frames, 30 FPS)
  - 360-degree rotation views (19 angles)
  - 3D point cloud export (PLY format, 16,000 points)
- ✅ **Integrated Script**: `run_all_visualization.sh` for one-command execution
- ✅ **Environment Fix**: torch_scatter compatibility for PyTorch 2.6.0 + CUDA 12.4

### Project Checklist
- [x] Code on GitHub
- [x] Comprehensive visualization pipeline
- [x] Automated execution scripts
- [ ] Camera-ready on arXiv
- [ ] Add links to data
- [ ] Add more detailed usage
