#!/bin/bash
# Extended training pipeline with comprehensive visualization
# Uses frame_jump=2 instead of 5 for more data (9000 frames vs 3600)
# Training time: ~12-24 hours

set -e

CONFIG="configs/markerless_mouse_nerf_extended.json"
OUTPUT_DIR="output/markerless_mouse_nerf_extended"
LOG_DIR="$OUTPUT_DIR/logs"
EPOCHS=100

echo "========================================"
echo "Extended Training Pipeline"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Frame jump: 2 (9000 training frames)"
echo "Start time: $(date)"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Step 1: Estimate up direction
echo "[Step 1/7] Estimating up direction..."
python3 estimate_up_direction.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step1_up_direction.log"

# Step 2: Calculate center and rotation
echo "[Step 2/7] Calculating center and rotation..."
python3 calculate_center_rotation.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step2_center_rotation.log"

# Step 3: Calculate crop indices
echo "[Step 3/7] Calculating crop indices..."
python3 calculate_crop_indices.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step3_crop_indices.log"

# Step 4: Write images to HDF5
echo "[Step 4/7] Writing images to HDF5..."
python3 write_images.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step4_write_images.log"

# Step 5: Copy to ZARR
echo "[Step 5/7] Copying to ZARR format..."
python3 copy_to_zarr.py \
    "$OUTPUT_DIR/images/images.h5" \
    "$OUTPUT_DIR/images/images.zarr" \
    2>&1 | tee "$LOG_DIR/step5_copy_zarr.log"

# Step 6: Train model
echo "[Step 6/7] Training model ($EPOCHS epochs)..."
echo "This will take approximately 12-24 hours..."
python3 train_script.py "$CONFIG" --epochs $EPOCHS 2>&1 | tee "$LOG_DIR/step6_training.log"

# Step 7: Evaluate model
echo "[Step 7/7] Evaluating model..."
python3 evaluate_model.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step7_evaluation.log"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "End time: $(date)"
echo ""

# Generate comprehensive visualizations
echo "========================================"
echo "Generating Visualizations"
echo "========================================"

# Export extended animation sequence (100 frames)
echo "Exporting 100-frame animation sequence..."
python3 export_animation_sequence.py \
    --config "$CONFIG" \
    --start_frame 0 \
    --num_frames 100 \
    --ply --npz --json \
    2>&1 | tee "$LOG_DIR/export_animation.log"

# Create 360 rotation
echo "Creating 360-degree rotation..."
python3 generate_360_rotation.py \
    --config "$CONFIG" \
    --frame 50 \
    --num_views 36 \
    2>&1 | tee "$LOG_DIR/rotation_360.log"

# Export Gaussians for key frames
echo "Exporting Gaussian parameters for key frames..."
for frame in 0 25 50 75 99; do
    python3 export_gaussian_full.py --config "$CONFIG" --frame $frame --format npz
    python3 export_gaussian_full.py --config "$CONFIG" --frame $frame --format ply_extended
done

# Create Rerun visualizations
echo "Creating Rerun .rrd files..."
python3 visualize_gaussian_rerun.py \
    "$OUTPUT_DIR/gaussians/gaussian_frame0000.npz" \
    --save "$OUTPUT_DIR/visualizations/gaussian_frame0000.rrd"

python3 visualize_gaussian_rerun.py \
    "$OUTPUT_DIR/animation/gaussians_npz/" \
    --sequence \
    --save "$OUTPUT_DIR/visualizations/animation_sequence.rrd"

# Create organized export package
echo "Creating organized export package..."
python3 create_organized_export.py \
    --config "$CONFIG" \
    --include-checkpoint \
    2>&1 | tee "$LOG_DIR/create_export.log"

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Export package: exports/markerless_mouse_nerf_extended_TIMESTAMP/"
echo ""
echo "Visualizations created:"
echo "  - 100-frame animation sequence"
echo "  - 36-view 360° rotation"
echo "  - Rerun .rrd files"
echo "  - Gaussian NPZ files for frames 0, 25, 50, 75, 99"
echo ""
echo "Total time: $(date)"
