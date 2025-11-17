#!/bin/bash

# Render full temporal sequence from trained model
# Usage: bash scripts/visualization/render_full_sequence.sh [CONFIG] [START_FRAME] [END_FRAME] [VIEW_NUM]

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Activate conda environment
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate splatter 2>/dev/null || true

# Default parameters
CONFIG="${1:-configs/baseline/markerless_mouse_nerf.json}"
START_FRAME="${2:-0}"
END_FRAME="${3:-3600}"
VIEW_NUM="${4:-0}"
OUTPUT_DIR="output/markerless_mouse_nerf/renders/full_sequence"

echo "========================================="
echo "Full Temporal Sequence Rendering"
echo "========================================="
echo "Config: $CONFIG"
echo "Frames: $START_FRAME to $END_FRAME ($(($END_FRAME - $START_FRAME)) frames)"
echo "View: $VIEW_NUM"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Calculate frame step for progress updates
TOTAL_FRAMES=$(($END_FRAME - $START_FRAME))
UPDATE_INTERVAL=$((TOTAL_FRAMES / 20))  # Update every 5%
if [ $UPDATE_INTERVAL -lt 10 ]; then
    UPDATE_INTERVAL=10
fi

echo "Starting render..."
echo ""

# Render frames
for ((frame=$START_FRAME; frame<$END_FRAME; frame++)); do
    # Progress update
    if [ $((($frame - $START_FRAME) % $UPDATE_INTERVAL)) -eq 0 ]; then
        PROGRESS=$((100 * ($frame - $START_FRAME) / $TOTAL_FRAMES))
        echo "  Progress: $PROGRESS% (frame $frame / $END_FRAME)"
    fi

    # Render frame
    python scripts/visualization/render_image.py "$CONFIG" $frame $VIEW_NUM \
        --out_fn "$OUTPUT_DIR/frame$(printf "%05d" $frame).png" \
        2>&1 | grep -E "(Error|Traceback)" || true
done

echo ""
echo "✓ Rendering complete: $TOTAL_FRAMES frames"
echo ""

# Create video
if command -v ffmpeg &> /dev/null; then
    echo "Creating video..."

    # Try MP4 first
    if ffmpeg -y -framerate 30 -i "$OUTPUT_DIR/frame%05d.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        "$OUTPUT_DIR/full_sequence.mp4" 2>&1 | grep -q "Unknown encoder"; then

        echo "  libx264 not available, trying alternative encoders..."

        # Try h264_nvenc (NVIDIA GPU encoder)
        if ffmpeg -y -framerate 30 -i "$OUTPUT_DIR/frame%05d.png" \
            -c:v h264_nvenc -preset fast -crf 18 \
            "$OUTPUT_DIR/full_sequence.mp4" 2>&1 | grep -q "Unknown encoder"; then

            # Fallback to GIF
            echo "  Creating GIF instead..."
            ffmpeg -y -framerate 30 -i "$OUTPUT_DIR/frame%05d.png" \
                -vf "fps=15,scale=576:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
                "$OUTPUT_DIR/full_sequence.gif" 2>&1 | tail -3

            if [ -f "$OUTPUT_DIR/full_sequence.gif" ]; then
                echo "  ✓ Created: $OUTPUT_DIR/full_sequence.gif"
            fi
        else
            echo "  ✓ Created: $OUTPUT_DIR/full_sequence.mp4 (NVENC)"
        fi
    else
        echo "  ✓ Created: $OUTPUT_DIR/full_sequence.mp4"
    fi
else
    echo "⚠ ffmpeg not found, skipping video creation"
fi

echo ""
echo "========================================="
echo "Rendering Complete!"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Total frames: $TOTAL_FRAMES"
ls -lh "$OUTPUT_DIR"/*.mp4 "$OUTPUT_DIR"/*.gif 2>/dev/null || echo "No video files created"
echo ""
