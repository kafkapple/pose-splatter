## Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance [NeurIPS 2025]
[![arXiv](https://img.shields.io/badge/arXiv-2505.18342-b31b1b.svg)](https://arxiv.org/pdf/2505.18342.pdf) 

- Authors: [Jack Goffinet*](https://scholar.google.com/citations?user=-oXW2RYAAAAJ&hl=en),  [Youngjo Min*](https://sites.google.com/view/youngjo-min),  [Carlo Tomasi](https://users.cs.duke.edu/~tomasi/), [David Carlson](https://carlson.pratt.duke.edu/) (* denotes equal contribution)
<div align="center">

![Teaser Image](assets/teaser.png)

</div>


Code for "Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance."


### Quick Start

#### Prerequisites
```bash
# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gsplat torchmetrics matplotlib seaborn pandas tabulate h5py zarr
```

#### Dataset Setup

This project uses the **DANNCE `markerless_mouse_1` dataset** for demonstration.

**Option 1: Demo Data (Quick Start)**
```bash
# Download demo videos
wget -O vids.zip https://tinyurl.com/DANNCEmm1vids
unzip vids.zip -d data/markerless_mouse_1_nerf/

# Download preprocessed data from Google Drive
# Includes: undistorted videos, 2D keypoints, silhouettes
# See: https://github.com/tqxli/dannce-pytorch
```

**Option 2: Full Training Dataset**
- Access via [Duke Research Data Repository](https://github.com/tqxli/dannce-pytorch)
- Follow instructions in the DANNCE-PyTorch repository

**Expected Data Structure**:
```
data/markerless_mouse_1_nerf/
├── videos_undist/          # 6-camera RGB videos (128MB)
│   ├── 0.mp4              # Camera 0: 1152×1024, 18K frames
│   ├── 1.mp4 ... 5.mp4    # Cameras 1-5
├── simpleclick_undist/     # Segmentation masks (64MB)
│   ├── 0.mp4 ... 5.mp4    # Binary masks for each camera
├── camera_params.h5        # Camera calibration (5KB)
└── keypoints2d_undist/     # 2D keypoint detections (55MB)
```

**Dataset Specifications**:
- **Cameras**: 6 synchronized views
- **Resolution**: 1152 × 1024 pixels
- **Frame Rate**: 100 FPS
- **Total Frames**: 18,000 (3 minutes)
- **Subject**: Mouse in natural behavior

**Citation**:
- Original DANNCE: [spoonsso/DANNCE](https://github.com/spoonsso/DANNCE)
- PyTorch Implementation: [tqxli/dannce-pytorch](https://github.com/tqxli/dannce-pytorch)

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

#### Supported Software

- **MeshLab**: PLY viewing and editing
- **CloudCompare**: Point cloud analysis
- **Blender**: Animation and rendering
- **Python**: NumPy/SciPy analysis with NPZ
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
