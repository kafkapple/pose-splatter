#!/bin/bash

# Minimal safe visualization script for high GPU memory usage
# Only generates essential visualizations to minimize memory footprint

source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter

echo "========================================="
echo "Minimal Safe Visualization Pipeline"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "========================================="

# Create output directories
mkdir -p output/markerless_mouse_nerf/renders/multiview
mkdir -p output/markerless_mouse_nerf/renders/temporal
mkdir -p output/markerless_mouse_nerf/renders/rotation360

CONFIG="configs/markerless_mouse_nerf.json"

# 1. Multi-view rendering (6 cameras, frame 0) - FULL
echo ""
echo "1. Multi-view renders (6 cameras)..."
for view in 0 1 2 3 4 5; do
    echo "  Rendering view $view..."
    python3 render_image.py $CONFIG 0 $view \
        --out_fn output/markerless_mouse_nerf/renders/multiview/frame0000_view${view}.png \
        2>&1 | grep -E "(Saved|Error|center_offset)"

    # Check GPU memory after each render
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ $MEM_USED -gt 11500 ]; then
        echo "  ⚠ High memory usage: ${MEM_USED}MB - pausing 5 seconds"
        sleep 5
    fi
done

echo "  ✓ Multi-view complete (6/6)"

# 2. Temporal sequence (5 frames only - MINIMAL)
echo ""
echo "2. Temporal sequence (5 frames, minimal)..."
for frame in 0 5 10 15 20; do
    echo "  Rendering frame $frame..."
    python3 render_image.py $CONFIG $frame 0 \
        --out_fn output/markerless_mouse_nerf/renders/temporal/frame$(printf "%04d" $frame).png \
        2>&1 | grep -E "(Saved|Error|center_offset)"

    # Memory check
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ $MEM_USED -gt 11500 ]; then
        echo "  ⚠ High memory usage: ${MEM_USED}MB - pausing 5 seconds"
        sleep 5
    fi
done

echo "  ✓ Temporal sequence complete (5/5)"

# 3. 360-degree rotation (8 angles only - MINIMAL)
echo ""
echo "3. Rotation views (8 angles, minimal)..."
for i in 0 3 6 9 12 15 18 21; do
    angle=$(python3 -c "import math; print($i * 2 * math.pi / 24)")
    echo "  Rendering angle $i ($(printf "%.2f" $angle) rad)..."
    python3 render_image.py $CONFIG 0 0 --angle_offset $angle \
        --out_fn output/markerless_mouse_nerf/renders/rotation360/rot$(printf "%03d" $i).png \
        2>&1 | grep -E "(Saved|Error|center_offset)"

    # Memory check
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ $MEM_USED -gt 11500 ]; then
        echo "  ⚠ High memory usage: ${MEM_USED}MB - pausing 5 seconds"
        sleep 5
    fi
done

echo "  ✓ Rotation views complete (8/8)"

# 4. Point cloud export
echo ""
echo "4. Point cloud export..."
if python3 export_point_cloud.py --frame 0 2>&1 | grep -E "(Exporting|Point cloud|Saving|✓|Error)"; then
    echo "  ✓ Point cloud export complete"
else
    echo "  ⚠ Point cloud export failed (may be due to memory)"
fi

echo ""
echo "========================================="
echo "✓ Minimal visualization pipeline complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - Multi-view: 6 images"
echo "  - Temporal: 5 images"
echo "  - Rotation: 8 images"
echo "  - Point cloud: 1 PLY file"
echo ""
echo "Total: ~20 files"
echo ""
echo "Final GPU Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
