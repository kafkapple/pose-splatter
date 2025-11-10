"""
Generate multi-viewpoint renders from all 6 camera angles.
"""
import os
import sys
from tqdm import tqdm

# Parameters
config_path = "configs/markerless_mouse_nerf.json"
frame_num = 0  # Can be changed
num_cameras = 6
output_dir = "output/markerless_mouse_nerf/renders/multiview"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print(f"Generating multi-view renders for frame {frame_num}")
print(f"Output directory: {output_dir}")

# Render from all camera viewpoints
for view_num in tqdm(range(num_cameras), desc="Rendering views"):
    output_fn = os.path.join(output_dir, f"frame{frame_num:04d}_view{view_num}.png")

    cmd = f"python3 render_image.py {config_path} {frame_num} {view_num} --out_fn {output_fn} 2>&1 | grep -E '(Saved|Error|center_offset)'"
    ret = os.system(cmd)

    if ret != 0:
        print(f"Warning: Failed to render view {view_num}")

print(f"\n✓ Multi-view rendering complete!")
print(f"  Generated {num_cameras} images in {output_dir}")
