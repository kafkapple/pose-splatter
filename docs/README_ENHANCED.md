# Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance [NeurIPS 2025]

[![arXiv](https://img.shields.io/badge/arXiv-2505.18342-b31b1b.svg)](https://arxiv.org/pdf/2505.18342.pdf)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**Authors**: [Jack Goffinet*](https://scholar.google.com/citations?user=-oXW2RYAAAAJ&hl=en), [Youngjo Min*](https://sites.google.com/view/youngjo-min), [Carlo Tomasi](https://users.cs.duke.edu/~tomasi/), [David Carlson](https://carlson.pratt.duke.edu/) (* denotes equal contribution)

<div align="center">

![Teaser Image](assets/teaser.png)

</div>

Code for "Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance."

---

## 🚀 Quick Start

**New users**: See [QUICKSTART.md](QUICKSTART.md) for 5-minute setup!

```bash
# 1. Install environment
conda env create -f environment.yml
conda activate pose-splatter

# 2. Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from src.model import PoseSplatter; print('Model: OK')"

# 3. (With data) Run pipeline
python estimate_up_direction.py config.json
python calculate_center_rotation.py config.json
python calculate_crop_indices.py config.json
python write_images.py config.json
python copy_to_zarr.py images/images.h5 images/images.zarr
python train_script.py config.json --epochs 50
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute setup guide |
| **[SETUP_GUIDE.md](SETUP_GUIDE.md)** | Detailed installation & troubleshooting |
| **[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** | Complete technical analysis & pipeline |
| [requirements.txt](requirements.txt) | Python dependencies |
| [environment.yml](environment.yml) | Conda environment |

---

## 📦 Installation

### Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate pose-splatter
```

### Option 2: Manual Installation

```bash
# Create environment
conda create -n pose-splatter python=3.10 -y
conda activate pose-splatter

# Install PyTorch with CUDA
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- NVIDIA GPU with CUDA support (8GB+ VRAM)
- CUDA 11.8
- Python 3.10

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for troubleshooting.

---

## 🎯 Usage Pipeline

### Step-by-Step Execution

```bash
# 1. Collect up direction
python estimate_up_direction.py config.json

# 2. Get center and rotations
python calculate_center_rotation.py config.json

# 3. Get volume sum, decide volume_idx
python calculate_crop_indices.py config.json

# 4. Write images in HDF5 format
python write_images.py config.json

# 5. Copy images to ZARR for training
python copy_to_zarr.py path/to/input.h5 path/to/output.zarr

# 6. Train a model
python train_script.py config.json

# 7. Evaluate model
python evaluate_model.py config.json

# 8. Render an image
python render_image.py config.json <frame_num> <view_num>

# 9. Calculate visual features
python calculate_visual_features.py config.json

# 10. Calculate visual embedding
python calculate_visual_embedding.py config.json
```

---

## 📊 Model Architecture

**Pose Splatter** combines:
- **Shape Carving**: Multi-view silhouette-based 3D reconstruction
- **3D U-Net**: Volumetric feature learning with skip connections
- **3D Gaussian Splatting**: Differentiable rendering with neural primitives

**Pipeline**:
```
Multi-view Images → Shape Carving → 3D U-Net → Gaussian Params → Rendering
     [C,H,W,3]         [4,n³]        [8,n³]         [N,14]        [H,W,3]
```

See [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) for detailed architecture.

---

## ⚙️ Configuration

Example config (`configs/mouse_4.json`):

```json
{
  "data_directory": "/path/to/data/mouse/",
  "project_directory": "/path/to/project/mouse_4_cameras/",
  "image_downsample": 4,
  "ell": 0.22,
  "grid_size": 112,
  "volume_idx": [[0, 96], [16, 96], [25, 89]],
  "lr": 1e-4,
  "img_lambda": 0.5,
  "ssim_lambda": 0.0
}
```

**Key Parameters**:
- `image_downsample`: Resolution reduction factor (default: 4)
- `ell`: Volume size in meters (default: 0.22)
- `grid_size`: Voxel resolution (default: 112)
- `volume_idx`: Crop indices for volume (determined by step 3)
- `lr`: Learning rate (default: 1e-4)

---

## 🧪 Training

### Basic Training

```bash
python train_script.py config.json --epochs 50
```

### Debug Mode (Fast Verification)

```bash
python train_script.py config.json --epochs 5 --max_batches 50
```

### Ablation Study

```bash
python train_script.py config.json --epochs 50 --ablation
```

**Outputs**:
- `reconstruction.pdf`: Predicted images (updated every epoch)
- `loss.pdf`: Training curves
- `checkpoint.pt`: Model weights

---

## 📈 Evaluation

```bash
# Quantitative evaluation
python evaluate_model.py config.json
# Output: metrics_test.csv (IoU, SSIM, PSNR, L1)

# Render single image
python render_image.py config.json 100 0

# Render with pose variation
python render_image.py config.json 100 0 --angle_offset 0.5 --delta_x 0.1
```

---

## 📂 Data Format

### Required Input Data

```
data_directory/
├── Camera1/0.mp4, Camera2/0.mp4, ...  # RGB videos
├── mask_videos/1.mp4, 2.mp4, ...      # Silhouette masks
└── camera_params_*.h5                  # Camera calibration
```

**Camera Calibration Format** (HDF5):
```
/camera_parameters/
  ├── rotation: [C, 3, 3]
  ├── translation: [C, 3]
  └── intrinsic: [C, 3, 3]
```

**Note**: Public datasets are not yet available. See issue [#TODO] for data requests.

---

## 🔧 Troubleshooting

### Common Issues

**CUDA out of memory**:
```json
// Reduce in config.json
"image_downsample": 8,  // 4 → 8
"grid_size": 64,        // 112 → 64
```

**gsplat installation fails**:
```bash
pip cache purge
pip install gsplat --no-cache-dir
```

**torch-scatter installation fails**:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete troubleshooting guide.

---

## 📖 Citation

```bibtex
@inproceedings{goffinet2025pose,
  title={Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance},
  author={Goffinet, Jack and Min, Youngjo and Tomasi, Carlo and Carlson, David E.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

## 🗺️ Project Roadmap

### Current Status

- [x] Code on GitHub
- [x] Installation guides (requirements.txt, environment.yml)
- [x] Comprehensive documentation (ANALYSIS_REPORT.md)
- [x] Quick start guide
- [ ] Camera-ready on arXiv
- [ ] Add links to data
- [ ] Public dataset release
- [ ] Tutorial notebooks

### Future Plans

- [ ] Docker image for easy deployment
- [ ] TensorBoard integration
- [ ] Real-time inference optimization
- [ ] Few-view extension
- [ ] Tutorial videos

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For major changes, please open an issue first to discuss.

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/[author]/pose-splatter/issues)
- **Authors**: See paper for contact information
- **Documentation**: See [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) for technical details

---

## 📄 License

See [LICENSE.md](LICENSE.md) for details.

---

## 🙏 Acknowledgments

- **gsplat**: 3D Gaussian Splatting implementation
- **PyTorch**: Deep learning framework
- **NeurIPS 2025**: For accepting our work

---

## 📚 Related Projects

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [SLEAP](https://sleap.ai/)
- [DeepLabCut](https://deeplabcut.github.io/DeepLabCut/)

---

**Made with ❤️ by the Pose Splatter team**

**Enhanced documentation created: 2025-11-09**
