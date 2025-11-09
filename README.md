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

### Project Checklist
- [x] Code on GitHub
- [ ] Camera-ready on arXiv
- [ ] Add links to data
- [ ] Add more detailed usage
