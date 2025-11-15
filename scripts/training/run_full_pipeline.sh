#!/bin/bash
# Full Pose Splatter Pipeline Execution Script
# Estimated total time: 8-12 hours

set -e  # Exit on error

CONFIG="configs/markerless_mouse_nerf.json"
LOG_DIR="output/markerless_mouse_nerf/logs"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Pose Splatter Full Pipeline"
echo "========================================="
echo "Config: $CONFIG"
echo "Start time: $(date)"
echo ""

# Step 1: Up direction (already done)
echo "[Step 1] Up direction estimation - SKIPPED (already completed)"
echo ""

# Step 2: Calculate center and rotation
echo "[Step 2] Calculate center and rotation - STARTING"
echo "  Estimated time: 1-2 hours"
echo "  Processing ~3,600 frames (frame_jump=5)"
python3 calculate_center_rotation.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step2_center_rotation.log"
echo "[Step 2] COMPLETED at $(date)"
echo ""

# Step 3: Calculate crop indices
echo "[Step 3] Calculate volume crop indices - STARTING"
echo "  Estimated time: 1-2 hours"
python3 calculate_crop_indices.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step3_crop_indices.log"
echo "[Step 3] COMPLETED at $(date)"
echo ""
echo "IMPORTANT: Check the output above for recommended volume_idx"
echo "Update config.json with the volume_idx before continuing"
read -p "Press Enter to continue after updating volume_idx..."

# Step 4: Write images to HDF5
echo "[Step 4] Write images to HDF5 - STARTING"
echo "  Estimated time: 2-4 hours"
echo "  Processing ~3,600 frames in parallel"
python3 write_images.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step4_write_images.log"
echo "[Step 4] COMPLETED at $(date)"
echo ""

# Step 5: Convert to Zarr
echo "[Step 5] Convert HDF5 to Zarr - STARTING"
echo "  Estimated time: 30-60 minutes"
INPUT_H5="output/markerless_mouse_nerf/images/images.h5"
OUTPUT_ZARR="output/markerless_mouse_nerf/images/images.zarr"
python3 copy_to_zarr.py "$INPUT_H5" "$OUTPUT_ZARR" 2>&1 | tee "$LOG_DIR/step5_zarr_conversion.log"
echo "[Step 5] COMPLETED at $(date)"
echo ""

# Step 6: Train model
echo "[Step 6] Train model - STARTING"
echo "  Estimated time: 4-8 hours (50 epochs)"
echo "  GPU will be heavily used"
python3 train_script.py "$CONFIG" --epochs 50 2>&1 | tee "$LOG_DIR/step6_training.log"
echo "[Step 6] COMPLETED at $(date)"
echo ""

# Step 7: Evaluate model
echo "[Step 7] Evaluate model - STARTING"
echo "  Estimated time: 30-60 minutes"
python3 evaluate_model.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step7_evaluation.log"
echo "[Step 7] COMPLETED at $(date)"
echo ""

# Step 8: Render sample images
echo "[Step 8] Render sample images - STARTING"
echo "  Rendering frames 100, 500, 1000 from all views"
for frame in 100 500 1000; do
    for view in 0 1 2 3 4 5; do
        echo "  Rendering frame $frame, view $view"
        python3 render_image.py "$CONFIG" $frame $view 2>&1 | tee -a "$LOG_DIR/step8_rendering.log"
    done
done
echo "[Step 8] COMPLETED at $(date)"
echo ""

echo "========================================="
echo "PIPELINE COMPLETED!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Model checkpoint: output/markerless_mouse_nerf/checkpoint.pt"
echo "  - Metrics: output/markerless_mouse_nerf/metrics_test.csv"
echo "  - Renderings: output/markerless_mouse_nerf/renders/"
echo "  - Logs: $LOG_DIR/"
echo ""
echo "Next steps:"
echo "  1. Check metrics_test.csv for performance"
echo "  2. View rendered images in renders/"
echo "  3. Run additional rendering with different poses"
