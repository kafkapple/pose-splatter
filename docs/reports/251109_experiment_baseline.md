# Pose Splatter Baseline Experiment

**Date**: 2025-11-09
**Experiment ID**: baseline_v1
**Status**: Running

---

## 📋 Overview

This is the baseline experiment for the Pose Splatter markerless mouse tracking and 3D reconstruction project.

### Objectives
- Establish baseline performance metrics for the Pose Splatter model
- Validate the full pipeline from data preprocessing to model evaluation
- Generate reference results for future experiments

---

## 🔧 Configuration

### Data
- **Dataset**: markerless_mouse_1_nerf
- **Video Resolution**: 1152 x 1024
- **Number of Cameras**: 6
- **Holdout Views**: [5, 1]
- **Frame Rate**: 100 fps
- **Frame Jump**: 5 (every 5th frame used)
- **Total Frames**: 18,000

### Model Parameters
- **Grid Size**: 112
- **Image Downsample**: 4x
- **ell (voxel size)**: 0.22
- **ell_tracking**: 0.25
- **Volume Fill Color**: 0.38
- **Adaptive Camera**: False

### Training Parameters
- **Learning Rate**: 1e-4
- **Epochs**: 50
- **Image Lambda**: 0.5
- **SSIM Lambda**: 0.0
- **Validation Frequency**: Every 5 epochs
- **Save Frequency**: Every 1 epoch

---

## 📊 Pipeline Steps

### Step 1: Data Preparation ✅
- Camera parameters loaded from HDF5
- 2D keypoints available for all 6 views
- Videos and mask videos verified

### Step 2: Calculate Center & Rotation 🔄
- **Status**: Running
- **Start Time**: 2025-11-09 13:54:29
- **Expected Duration**: ~5-10 minutes
- **Processing**: 18,000 frames with parallel processing

### Step 3: Calculate Crop Indices ⏳
- **Status**: Pending
- **Expected Duration**: ~1 minute

### Step 4: Write Images to HDF5 ⏳
- **Status**: Pending
- **Expected Duration**: 2-4 hours
- **Note**: This is the most time-consuming preprocessing step

### Step 5: Convert to Zarr ⏳
- **Status**: Pending
- **Expected Duration**: ~30 minutes

### Step 6: Train Model ⏳
- **Status**: Pending
- **Expected Duration**: 4-8 hours
- **Epochs**: 50
- **GPU**: NVIDIA GeForce RTX 3060 (12GB)

### Step 7: Evaluate Model ⏳
- **Status**: Pending
- **Expected Duration**: ~10 minutes

### Step 8: Render Sample Images ⏳
- **Status**: Pending
- **Frames**: 100, 500, 1000
- **Views**: 0, 2, 4

---

## 💾 Output Files

### Expected Outputs
- `output/markerless_mouse_nerf/center_rotation.npz` - Center and rotation data
- `output/markerless_mouse_nerf/images/images.h5` - Preprocessed images (HDF5)
- `output/markerless_mouse_nerf/images/images.zarr` - Preprocessed images (Zarr)
- `output/markerless_mouse_nerf/checkpoint.pt` - Trained model checkpoint
- `output/markerless_mouse_nerf/metrics_test.csv` - Evaluation metrics
- `output/markerless_mouse_nerf/renders/` - Rendered sample images

### Log Files
All logs are saved to: `output/markerless_mouse_nerf/logs/`
- `step2_center_rotation.log`
- `step3_crop_indices.log`
- `step4_write_images.log`
- `step5_zarr.log`
- `step6_training.log`
- `step7_evaluation.log`
- `step8_rendering.log`

---

## 📈 Expected Metrics

Based on the evaluation script, the following metrics will be computed:
- **L1 Loss**: Mean absolute error
- **IoU**: Intersection over Union (hard threshold)
- **Soft IoU**: Soft IoU with alpha blending
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

Metrics will be computed for:
- Train split (first 1/3 of frames)
- Validation split (middle 1/3 of frames)
- Test split (last 1/3 of frames)

---

## 🔍 Monitoring

### Real-time Monitoring
```bash
# Watch pipeline progress (updates every 10 seconds)
watch -n 10 ./monitor_pipeline.sh

# Check GPU usage
watch -n 2 nvidia-smi

# View specific log
tail -f output/markerless_mouse_nerf/logs/step6_training.log
```

### System Resources
- **GPU**: RTX 3060 (12GB VRAM)
- **CPU**: Multi-core (parallel processing enabled)
- **Storage**: ~500GB for full pipeline outputs

---

## 📝 Notes

### Experimental Conditions
- Using default hyperparameters from original Pose Splatter implementation
- No SSIM loss component (ssim_lambda = 0.0)
- Standard image downsampling of 4x
- Frame jump of 5 to reduce computational cost

### Potential Issues
- Long preprocessing time (Step 4: 2-4 hours)
- High storage requirements for HDF5/Zarr files
- GPU memory constraints during training

### Future Experiments Planned
1. **High Resolution** (configs/markerless_mouse_nerf_high_res.json)
   - image_downsample: 2x (vs 4x)
   - grid_size: 128 (vs 112)

2. **Fast Variant** (configs/markerless_mouse_nerf_fast.json)
   - image_downsample: 8x
   - grid_size: 64
   - frame_jump: 10
   - lr: 2e-4

3. **SSIM Loss** (configs/markerless_mouse_nerf_ssim.json)
   - img_lambda: 0.3
   - ssim_lambda: 0.2

---

## 📚 References

- Pose Splatter Paper: [TBD]
- Original Repository: [TBD]
- Dataset: markerless_mouse_1_nerf

---

## ✅ Completion Checklist

- [x] Environment setup verified
- [x] Data preparation completed
- [x] Configuration validated
- [x] Pipeline launched
- [ ] Center & rotation calculated
- [ ] Images written to HDF5
- [ ] Zarr conversion complete
- [ ] Model training complete
- [ ] Evaluation metrics generated
- [ ] Results analyzed
- [ ] Visualizations created

---

**Last Updated**: 2025-11-09 13:54 KST
