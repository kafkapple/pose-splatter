#!/bin/bash

# Monitor visualization progress

echo "Visualization Progress Monitor"
echo "=============================="
echo ""

# Function to count files in a directory
count_files() {
    dir=$1
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo $count
    else
        echo "0"
    fi
}

# Check progress
while true; do
    clear
    echo "=========================================="
    echo "Pose Splatter Visualization Progress"
    echo "=========================================="
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Multi-view (6 expected)
    multiview_count=$(count_files "output/markerless_mouse_nerf/renders/multiview")
    echo "1. Multi-view renders: $multiview_count / 6"

    # Temporal (30 expected)
    temporal_count=$(count_files "output/markerless_mouse_nerf/renders/temporal")
    echo "2. Temporal frames: $temporal_count / 31 (30 frames + 1 video)"

    # Rotation (24 expected)
    rotation_count=$(count_files "output/markerless_mouse_nerf/renders/rotation360")
    echo "3. Rotation frames: $rotation_count / 25 (24 frames + 1 video)"

    # Point cloud (1 expected)
    pointcloud_count=$(count_files "output/markerless_mouse_nerf/pointclouds")
    echo "4. Point clouds: $pointcloud_count / 1"

    echo ""
    echo "=========================================="
    echo "Latest log entries:"
    echo "=========================================="
    tail -10 output/markerless_mouse_nerf/logs/vis_full_pipeline.log 2>/dev/null || echo "Log file not yet created"

    echo ""
    echo "Press Ctrl+C to stop monitoring"

    sleep 5
done
