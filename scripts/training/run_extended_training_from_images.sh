#!/bin/bash
# Continue extended training from write_images step
# Run this after write_images.py completes

set -e

CONFIG="configs/markerless_mouse_nerf_extended.json"
OUTPUT_DIR="output/markerless_mouse_nerf_extended"
LOG_DIR="$OUTPUT_DIR/logs"
EPOCHS=100

echo "========================================"
echo "Extended Training (from images)"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Start time: $(date)"
echo ""

# Wait for write_images to complete
echo "Checking if write_images completed..."
while ! grep -q "Done writing images" "$LOG_DIR/write_images.log" 2>/dev/null; do
    echo "  Waiting for write_images... $(date +%H:%M:%S)"
    sleep 30
done
echo "✓ write_images completed!"

# Step 5: Copy to ZARR
echo "[Step 5/7] Copying to ZARR format..."
python3 copy_to_zarr.py \
    "$OUTPUT_DIR/images/images.h5" \
    "$OUTPUT_DIR/images/images.zarr" \
    2>&1 | tee "$LOG_DIR/step5_copy_zarr.log"

# Step 6: Train model
echo "[Step 6/7] Training model ($EPOCHS epochs)..."
echo "This will take approximately 16-30 hours (2.5x more data than baseline)..."
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

# Export 100-frame animation sequence
echo "Exporting 100-frame animation sequence..."
python3 export_animation_sequence.py \
    --config "$CONFIG" \
    --start_frame 0 \
    --num_frames 100 \
    --ply --npz \
    2>&1 | tee "$LOG_DIR/export_animation.log"

# Create Rerun visualizations
echo "Creating Rerun .rrd files..."
mkdir -p "$OUTPUT_DIR/visualizations"

python3 visualize_gaussian_rerun.py \
    "$OUTPUT_DIR/gaussians/gaussian_frame0000.npz" \
    --save "$OUTPUT_DIR/visualizations/gaussian_frame0000.rrd" \
    2>&1 | tee "$LOG_DIR/rerun_single.log"

python3 visualize_gaussian_rerun.py \
    "$OUTPUT_DIR/animation/gaussians_npz/" \
    --sequence \
    --save "$OUTPUT_DIR/visualizations/animation_100frames.rrd" \
    2>&1 | tee "$LOG_DIR/rerun_animation.log"

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
echo ""
echo "Visualizations created:"
echo "  - 100-frame animation sequence (.npz + .ply)"
echo "  - Rerun .rrd files (single frame + animation)"
echo "  - Export package with all results"
echo ""
echo "Training data: 9,000 frames (2.5x baseline)"
echo "Total time: $(date)"
