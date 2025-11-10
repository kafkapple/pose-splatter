"""
Generate 360-degree rotation views of a single frame.
"""
import os
import subprocess
import numpy as np
from tqdm import tqdm

# Parameters
config_path = "configs/markerless_mouse_nerf.json"
frame_num = 0
view_num = 0  # Base camera angle
num_angles = 36  # Number of rotation angles (10 degrees each)
fps = 30
output_dir = "output/markerless_mouse_nerf/renders/rotation360"
video_name = f"rotation360_frame{frame_num}.mp4"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print(f"Generating 360-degree rotation for frame {frame_num}")
print(f"Number of angles: {num_angles}")
print(f"Output directory: {output_dir}")

# Generate rotation angles (in radians)
angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

# Render at each rotation angle
for idx, angle in enumerate(tqdm(angles, desc="Rendering rotations")):
    output_fn = os.path.join(output_dir, f"rot{idx:03d}.png")

    cmd = f"conda run -n splatter python3 render_image.py {config_path} {frame_num} {view_num} --angle_offset {angle:.6f} --out_fn {output_fn} 2>&1 | grep -E '(Saved|Error)'"
    ret = os.system(cmd)

    if ret != 0:
        print(f"Warning: Failed to render angle {idx} ({np.degrees(angle):.1f}°)")

print(f"\n✓ Rotation rendering complete!")

# Create video using ffmpeg
print(f"Creating 360° rotation video with ffmpeg...")
video_path = os.path.join(output_dir, video_name)

ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-i", f"{output_dir}/rot%03d.png",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    video_path
]

try:
    result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    print(f"✓ Video created: {video_path}")
    print(f"  Duration: {num_angles/fps:.2f} seconds at {fps} fps")
    print(f"  Rotation: 360 degrees in {num_angles} steps")
except subprocess.CalledProcessError as e:
    print(f"Error creating video: {e}")
    print(f"STDERR: {e.stderr}")
except FileNotFoundError:
    print("Error: ffmpeg not found. Please install ffmpeg:")
    print("  sudo apt install ffmpeg")
