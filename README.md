# Pose Splatter: 3D Gaussian Splatting for Animal Pose and Appearance

[![arXiv](https://img.shields.io/badge/arXiv-2505.18342-b31b1b.svg)](https://arxiv.org/pdf/2505.18342.pdf)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)

**Authors:** [Jack Goffinet*](https://scholar.google.com/citations?user=-oXW2RYAAAAJ&hl=en), [Youngjo Min*](https://sites.google.com/view/youngjo-min), [Carlo Tomasi](https://users.cs.duke.edu/~tomasi/), [David Carlson](https://carlson.pratt.duke.edu/) (* equal contribution)

<div align="center">
<img src="assets/teaser.png" alt="Pose Splatter Teaser" width="800"/>
</div>

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation & Visualization](#evaluation--visualization)
- [Configuration Templates](#configuration-templates)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### Prerequisites

- **GPU**: 12GB+ (RTX 3060/3070 for 3D GS, 24GB+ A6000/4090 for 2D GS)
- **Python**: 3.10+
- **CUDA**: 11.8 or 12.8+

### Installation

```bash
# Create conda environment
conda create -n splatter python=3.10 -y
conda activate splatter

# Install PyTorch with CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install "numpy<2.0"
pip install gsplat torchmetrics matplotlib seaborn pandas tabulate h5py zarr
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Optional: Rerun for 3D visualization
pip install rerun-sdk

# Fix environment issues (if needed)
bash scripts/utils/fix_environment.sh
```

---

## Dataset Setup

This project uses the **DANNCE `markerless_mouse_1`** dataset preprocessed by MAMMAL_mouse.

**Option 1: Copy from MAMMAL_mouse repository**

```bash
mkdir -p data/markerless_mouse_1_nerf
cp -r /path/to/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \
      data/markerless_mouse_1_nerf/
```

**Option 2: Download directly**

```bash
# Download link and instructions
# See: https://github.com/MAMMAL_mouse_repo
```

**Verify dataset**:
```bash
python scripts/utils/verify_mammal_data.py
```

---

## Preprocessing

Run the full preprocessing pipeline (one-time setup):

```bash
bash scripts/preprocessing/run_full_preprocessing.sh \
  configs/baseline/markerless_mouse_nerf.json
```

**Pipeline steps**:
1. Camera parameter conversion (PKL → HDF5)
2. Up direction estimation
3. Center & rotation calculation
4. Crop indices calculation
5. Image conversion (HDF5 → ZARR)

**Expected time**: 10-30 minutes

**Output**: `output/markerless_mouse_nerf/` with `images/`, `volumes/`, `camera_params.h5`, etc.

---

## Training

### Quick Start

```bash
# Use appropriate config template for your GPU
bash scripts/training/run_training.sh \
  configs/templates/rtx3060_3d.json --epochs 50
```

### Config Templates

| Template | GPU | Mode | Resolution | Time | PSNR |
|----------|-----|------|------------|------|------|
| `rtx3060_3d.json` | 12GB | 3D GS | 288×256 | 6-8h | 25-27 dB |
| `a6000_2d.json` | 24GB+ | 2D GS | 576×512 | 10-15h | 28-30 dB |
| `debug_quick.json` | 4GB+ | 3D GS | 192×171 | 5-10m | N/A |

See [configs/templates/README.md](configs/templates/README.md) for details.

### Background Training

```bash
nohup bash scripts/training/run_training.sh \
  configs/templates/a6000_2d.json --epochs 50 \
  > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Monitor Training

```bash
# GPU usage
watch -n 2 nvidia-smi

# Training logs
tail -f output/markerless_mouse_nerf/logs/step6_training.log
```

---

## Evaluation & Visualization

### Evaluation

Compute metrics (PSNR, SSIM, LPIPS) on holdout views:

```bash
bash scripts/training/run_evaluation.sh \
  configs/baseline/markerless_mouse_nerf.json

# Results: output/markerless_mouse_nerf/evaluation_metrics.json
```

### Visualization

#### Quick Visualization (30 frames)

```bash
bash scripts/visualization/run_all_visualization.sh
```

**Outputs**:
- Multi-view renders (6 cameras)
- Temporal sequence (30 frames + video)
- 360° rotation (24 angles + video)
- 3D point cloud export

#### Full Sequence (3600 frames)

```bash
# Render all frames
bash scripts/visualization/render_full_sequence.sh \
  configs/baseline/markerless_mouse_nerf.json 0 3600 0
```

#### Rerun Interactive Visualization

```bash
# Export sequence with Rerun
python scripts/visualization/export_temporal_sequence_rerun.py \
  configs/baseline/markerless_mouse_nerf.json \
  --start 0 --end 3600 --view 0

# View in Rerun
rerun output/markerless_mouse_nerf/renders/full_sequence/sequence.rrd
```

**Features**:
- Interactive 3D Gaussian visualization
- Timeline scrubbing
- Multiple camera views
- Real-time parameter adjustment

#### Create Videos

```bash
# Create MP4/GIF from rendered frames
bash scripts/visualization/create_videos.sh
```

---

## Configuration Templates

### GPU-Specific Settings

**RTX 3060/3070 (12GB) - 3D GS**:
```json
{
  "image_downsample": 4,
  "grid_size": 112,
  "gaussian_mode": "3d",
  "gaussian_config": {}
}
```

**A6000/4090 (24GB+) - 2D GS**:
```json
{
  "image_downsample": 2,
  "grid_size": 128,
  "gaussian_mode": "2d",
  "gaussian_config": {
    "sigma_cutoff": 3.0,
    "kernel_size": 5,
    "batch_size": 5
  },
  "ssim_lambda": 0.1
}
```

### Key Parameters

| Parameter | Values | Effect |
|-----------|--------|--------|
| `image_downsample` | 2, 4, 6, 8 | Lower = higher quality, more memory |
| `grid_size` | 64, 112, 128, 256 | Higher = finer details, more memory |
| `gaussian_mode` | "2d", "3d" | 2D = higher quality, 3D = faster |
| `frame_jump` | 2, 5, 10 | Higher = fewer frames, faster training |

See [docs/reports/CONFIGURATION_GUIDE.md](docs/reports/CONFIGURATION_GUIDE.md) for full details.

---

## Troubleshooting

### Common Issues

**1. NumPy Version Conflict**
```bash
# Error: "NumPy 1.x cannot be run in NumPy 2.0.1"
bash scripts/utils/fix_environment.sh
```

**2. ModuleNotFoundError: torch_scatter**
```bash
bash scripts/utils/fix_environment.sh
```

**3. CUDA Out of Memory**

Reduce memory usage in config:
```json
{
  "image_downsample": 6,  // Increase (4 → 6)
  "grid_size": 96,         // Decrease (112 → 96)
  "gaussian_config": {
    "batch_size": 3        // Decrease (5 → 3, 2D only)
  }
}
```

**4. Import Errors on Different Servers**
```bash
# Always use wrapper scripts (handles PYTHONPATH)
bash scripts/training/run_training.sh CONFIG
```

**Complete troubleshooting**: [docs/troubleshooting/environment_errors.md](docs/troubleshooting/environment_errors.md)

---

## Advanced Usage

### Custom Configuration

```python
import json

# Load template
with open('configs/templates/rtx3060_3d.json') as f:
    config = json.load(f)

# Modify
config['project_directory'] = 'output/my_experiment/'
config['lr'] = 5e-5
config['ssim_lambda'] = 0.1

# Save
with open('configs/experiments/my_custom.json', 'w') as f:
    json.dump(config, f, indent=4)
```

### Blender Integration

Export 3D Gaussians and import into Blender:

```bash
# Export point cloud
python scripts/visualization/export_point_cloud.py --frame 0

# Import PLY file in Blender
# File → Import → Stanford PLY
```

See [scripts/visualization/blender_import_pointcloud.py](scripts/visualization/blender_import_pointcloud.py) for automated import.

### Batch Processing

```bash
# Process multiple experiments
for config in configs/experiments/*.json; do
    bash scripts/training/run_training.sh "$config" --epochs 50
    bash scripts/training/run_evaluation.sh "$config"
done
```

---

## Documentation

- **Preprocessing Guide**: [docs/reports/PREPROCESSING_GUIDE_BEGINNER.md](docs/reports/PREPROCESSING_GUIDE_BEGINNER.md)
- **Configuration Guide**: [docs/reports/CONFIGURATION_GUIDE.md](docs/reports/CONFIGURATION_GUIDE.md)
- **Camera Coordinate Systems**: [docs/reports/CAMERA_COORDINATE_SYSTEMS.md](docs/reports/CAMERA_COORDINATE_SYSTEMS.md)
- **Troubleshooting**: [docs/troubleshooting/environment_errors.md](docs/troubleshooting/environment_errors.md)
- **Template Guide**: [configs/templates/README.md](configs/templates/README.md)

---

## Citation

```bibtex
@inproceedings{goffinet2025posesplatter,
  title={Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance},
  author={Goffinet, Jack and Min, Youngjo and Tomasi, Carlo and Carlson, David},
  booktitle={NeurIPS},
  year={2025}
}
```

---

## License

[Add your license information here]

---

## Acknowledgments

This project builds upon:
- [DANNCE](https://github.com/spoonsso/dannce) - Multi-camera 3D pose estimation
- [MAMMAL](https://github.com/cfos3120/MAMMAL) - Mouse dataset preprocessing
- [gsplat](https://github.com/nerfstudio-project/gsplat) - 3D Gaussian Splatting
- [Rerun](https://rerun.io/) - Interactive 3D visualization
