#!/bin/bash

# Render 30 temporal frames for longer video

source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter

echo "========================================"
echo "Rendering 30 Temporal Frames"
echo "========================================"
echo ""

CONFIG="configs/markerless_mouse_nerf.json"
OUTPUT_DIR="output/markerless_mouse_nerf/renders/temporal"

mkdir -p $OUTPUT_DIR

# Render frames 0-29
for frame in {0..29}; do
    output_file="${OUTPUT_DIR}/frame$(printf "%04d" $frame).png"

    if [ -f "$output_file" ]; then
        echo "Frame $frame already exists, skipping..."
    else
        echo "Rendering frame $frame..."
        python3 render_image.py $CONFIG $frame 0 --out_fn $output_file 2>&1 | grep -E "(Saved|Error|center_offset)"
    fi
done

echo ""
echo "========================================"
echo "Rendering Complete!"
echo "========================================"
echo ""
echo "Total frames: $(ls -1 ${OUTPUT_DIR}/frame*.png 2>/dev/null | wc -l)"
echo ""

# Create video
echo "Creating video..."
find ${OUTPUT_DIR} -name "frame*.png" -type f | sort -V | awk '{printf "file '\''%s'\''\n", $0}' > /tmp/temporal_long_filelist.txt

ffmpeg -y -f concat -safe 0 -r 30 -i /tmp/temporal_long_filelist.txt \
    -c:v libopenh264 -pix_fmt yuv420p \
    ${OUTPUT_DIR}/temporal_long.mp4 2>&1 | tail -10

echo ""
echo "Video saved: ${OUTPUT_DIR}/temporal_long.mp4"
ls -lh ${OUTPUT_DIR}/temporal_long.mp4
