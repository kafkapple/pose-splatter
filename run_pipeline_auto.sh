#!/bin/bash
# Automated Full Pipeline (no manual intervention needed)
# This version auto-determines volume_idx

set -e

CONFIG="configs/markerless_mouse_nerf.json"
LOG_DIR="output/markerless_mouse_nerf/logs"
PROJECT_DIR="output/markerless_mouse_nerf"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "Pose Splatter Automated Full Pipeline"
echo "========================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config: $CONFIG"
echo ""

# Step 2: Calculate center and rotation
echo "[Step 2/7] Calculate center and rotation"
echo "  Start: $(date '+%H:%M:%S')"
python3 calculate_center_rotation.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step2_center_rotation.log"
echo "  Done: $(date '+%H:%M:%S')"
echo ""

# Step 3: Calculate crop indices
echo "[Step 3/7] Calculate volume crop indices"
echo "  Start: $(date '+%H:%M:%S')"
python3 calculate_crop_indices.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step3_crop_indices.log"

# Auto-extract volume_idx from output (last occurrence of pattern like [[x,y],[a,b],[c,d]])
VOLUME_IDX=$(grep -oP '\[\[.*?\]\]' "$LOG_DIR/step3_crop_indices.log" | tail -1)
if [ ! -z "$VOLUME_IDX" ]; then
    echo "  Auto-detected volume_idx: $VOLUME_IDX"
    # Update config.json
    python3 << EOF
import json
with open('$CONFIG', 'r') as f:
    config = json.load(f)
config['volume_idx'] = $VOLUME_IDX
with open('$CONFIG', 'w') as f:
    json.dump(config, f, indent=4)
print("  Updated config.json with volume_idx")
EOF
else
    echo "  WARNING: Could not auto-detect volume_idx, using default"
fi
echo "  Done: $(date '+%H:%M:%S')"
echo ""

# Step 4: Write images to HDF5
echo "[Step 4/7] Write images to HDF5"
echo "  Start: $(date '+%H:%M:%S')"
echo "  This will take 2-4 hours..."
python3 write_images.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step4_write_images.log"
echo "  Done: $(date '+%H:%M:%S')"
echo ""

# Step 5: Convert to Zarr
echo "[Step 5/7] Convert HDF5 to Zarr"
echo "  Start: $(date '+%H:%M:%S')"
INPUT_H5="$PROJECT_DIR/images/images.h5"
OUTPUT_ZARR="$PROJECT_DIR/images/images.zarr"
python3 copy_to_zarr.py "$INPUT_H5" "$OUTPUT_ZARR" 2>&1 | tee "$LOG_DIR/step5_zarr.log"
echo "  Done: $(date '+%H:%M:%S')"
echo ""

# Step 6: Train model
echo "[Step 6/7] Train model (50 epochs)"
echo "  Start: $(date '+%H:%M:%S')"
echo "  This will take 4-8 hours..."
python3 train_script.py "$CONFIG" --epochs 50 2>&1 | tee "$LOG_DIR/step6_training.log"
echo "  Done: $(date '+%H:%M:%S')"
echo ""

# Step 7: Evaluate model
echo "[Step 7/7] Evaluate model"
echo "  Start: $(date '+%H:%M:%S')"
python3 evaluate_model.py "$CONFIG" 2>&1 | tee "$LOG_DIR/step7_evaluation.log"
echo "  Done: $(date '+%H:%M:%S')"
echo ""

# Render samples
echo "[Bonus] Render sample images"
echo "  Start: $(date '+%H:%M:%S')"
for frame in 100 500 1000; do
    for view in 0 2 4; do
        python3 render_image.py "$CONFIG" $frame $view 2>&1 >> "$LOG_DIR/step8_rendering.log"
    done
done
echo "  Done: $(date '+%H:%M:%S')"
echo ""

echo "========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results:"
echo "  - Checkpoint: $PROJECT_DIR/checkpoint.pt"
echo "  - Metrics: $PROJECT_DIR/metrics_test.csv"
echo "  - Renders: $PROJECT_DIR/renders/"
echo "  - All logs: $LOG_DIR/"
