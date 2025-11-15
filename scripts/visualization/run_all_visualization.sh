#!/bin/bash

# Activate conda environment
source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter

echo "========================================="
echo "Pose Splatter Visualization Pipeline"
echo "========================================="
echo ""

# Create output directories
mkdir -p output/markerless_mouse_nerf/renders/multiview
mkdir -p output/markerless_mouse_nerf/renders/temporal
mkdir -p output/markerless_mouse_nerf/renders/rotation360
mkdir -p output/markerless_mouse_nerf/pointclouds

CONFIG="configs/markerless_mouse_nerf.json"

# 1. Multi-view rendering (6 cameras, frame 0)
echo "1. Generating multi-view renders (6 cameras)..."
for view in 0 1 2 3 4 5; do
    echo "  - Rendering view $view"
    python3 render_image.py $CONFIG 0 $view \
        --out_fn output/markerless_mouse_nerf/renders/multiview/frame0000_view${view}.png \
        2>&1 | grep -E "(Saved|Error)"
done
echo "✓ Multi-view rendering complete"
echo ""

# 2. Temporal sequence (first 30 frames, view 0)
echo "2. Generating temporal sequence (30 frames)..."
for frame in {0..29}; do
    printf "  - Rendering frame %d\n" $frame
    python3 render_image.py $CONFIG $frame 0 \
        --out_fn output/markerless_mouse_nerf/renders/temporal/frame$(printf "%04d" $frame).png \
        2>&1 | grep -E "(Saved|Error)"
done

# Create video from frames
if command -v ffmpeg &> /dev/null; then
    echo "  - Creating video..."
    ffmpeg -y -framerate 30 -i output/markerless_mouse_nerf/renders/temporal/frame%04d.png \
           -c:v libx264 -pix_fmt yuv420p -crf 18 \
           output/markerless_mouse_nerf/renders/temporal/temporal_sequence.mp4 \
           2>&1 | tail -5
    echo "✓ Temporal video created"
else
    echo "⚠ ffmpeg not found, skipping video creation"
fi
echo ""

# 3. 360-degree rotation (24 angles)
echo "3. Generating 360-degree rotation (24 angles)..."
for i in {0..23}; do
    angle=$(python3 -c "import math; print($i * 2 * math.pi / 24)")
    printf "  - Rendering angle %d (%.2f radians)\n" $i $angle
    python3 render_image.py $CONFIG 0 0 --angle_offset $angle \
        --out_fn output/markerless_mouse_nerf/renders/rotation360/rot$(printf "%03d" $i).png \
        2>&1 | grep -E "(Saved|Error)"
done

# Create rotation video
if command -v ffmpeg &> /dev/null; then
    echo "  - Creating rotation video..."
    ffmpeg -y -framerate 30 -i output/markerless_mouse_nerf/renders/rotation360/rot%03d.png \
           -c:v libx264 -pix_fmt yuv420p -crf 18 \
           output/markerless_mouse_nerf/renders/rotation360/rotation360.mp4 \
           2>&1 | tail -5
    echo "✓ Rotation video created"
else
    echo "⚠ ffmpeg not found, skipping video creation"
fi
echo ""

# 4. Export point cloud
echo "4. Exporting 3D point cloud..."
python3 export_point_cloud.py --frame 0 2>&1 | grep -E "(Exporting|Point cloud|Saving|✓)"
echo ""

echo "========================================="
echo "Visualization Pipeline Complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - Multi-view: output/markerless_mouse_nerf/renders/multiview/"
echo "  - Temporal: output/markerless_mouse_nerf/renders/temporal/"
echo "  - Rotation: output/markerless_mouse_nerf/renders/rotation360/"
echo "  - Point cloud: output/markerless_mouse_nerf/pointclouds/"
