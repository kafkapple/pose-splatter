#!/bin/bash

# Create videos from rendered PNG sequences
# Supports both MP4 (with libx264) and GIF fallback

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================"
echo "Creating Videos from Renders"
echo "================================"
echo ""

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg not found. Please install ffmpeg."
    exit 1
fi

# Function to create video with fallback
create_video() {
    local input_pattern=$1
    local output_mp4=$2
    local output_gif=$3
    local framerate=${4:-30}
    local name=$5

    echo "Creating $name..."

    # Try MP4 with libx264
    if ffmpeg -y -framerate $framerate -i "$input_pattern" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        "$output_mp4" 2>&1 | grep -q "Unknown encoder"; then

        echo "  libx264 not available, creating GIF instead..."

        # Fallback to GIF
        ffmpeg -y -framerate $framerate -i "$input_pattern" \
            -vf "fps=15,scale=576:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
            "$output_gif" 2>&1 | tail -3

        if [ -f "$output_gif" ]; then
            echo "  ✓ Created: $output_gif"
        else
            echo "  ✗ Failed to create $name"
        fi
    else
        # MP4 creation successful (run again without grep)
        ffmpeg -y -framerate $framerate -i "$input_pattern" \
            -c:v libx264 -pix_fmt yuv420p -crf 18 \
            "$output_mp4" 2>&1 | tail -3

        if [ -f "$output_mp4" ]; then
            echo "  ✓ Created: $output_mp4"
        else
            echo "  ✗ Failed to create $name"
        fi
    fi
}

# Temporal sequence
if [ -d "output/markerless_mouse_nerf/renders/temporal" ] && \
   [ "$(ls output/markerless_mouse_nerf/renders/temporal/frame*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    create_video \
        "output/markerless_mouse_nerf/renders/temporal/frame%04d.png" \
        "output/markerless_mouse_nerf/renders/temporal/temporal_sequence.mp4" \
        "output/markerless_mouse_nerf/renders/temporal/temporal_sequence.gif" \
        30 \
        "Temporal sequence"
fi

# 360 rotation
if [ -d "output/markerless_mouse_nerf/renders/rotation360" ] && \
   [ "$(ls output/markerless_mouse_nerf/renders/rotation360/rot*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    create_video \
        "output/markerless_mouse_nerf/renders/rotation360/rot%03d.png" \
        "output/markerless_mouse_nerf/renders/rotation360/rotation360.mp4" \
        "output/markerless_mouse_nerf/renders/rotation360/rotation360.gif" \
        24 \
        "360° rotation"
fi

echo ""
echo "================================"
echo "Video creation complete!"
echo "================================"
