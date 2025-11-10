"""
Generate temporal sequence video from consecutive frames.
"""
import os
import subprocess
from tqdm import tqdm

# Parameters
config_path = "configs/markerless_mouse_nerf.json"
view_num = 0  # Camera angle
start_frame = 0
num_frames = 60  # Number of frames to render (1 second at 60fps)
fps = 30
output_dir = "output/markerless_mouse_nerf/renders/temporal"
video_name = f"temporal_view{view_num}_frames{start_frame}-{start_frame+num_frames}.mp4"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print(f"Generating temporal sequence for view {view_num}")
print(f"Frames: {start_frame} to {start_frame + num_frames - 1}")
print(f"Output directory: {output_dir}")

# Render consecutive frames
for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
    frame_num = start_frame + frame_idx
    output_fn = os.path.join(output_dir, f"frame{frame_num:04d}.png")

    cmd = f"conda run -n splatter python3 render_image.py {config_path} {frame_num} {view_num} --out_fn {output_fn} 2>&1 | grep -E '(Saved|Error)'"
    ret = os.system(cmd)

    if ret != 0:
        print(f"Warning: Failed to render frame {frame_num}")

print(f"\n✓ Frame rendering complete!")

# Create video using ffmpeg
print(f"Creating video with ffmpeg...")
video_path = os.path.join(output_dir, video_name)

ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-pattern_type", "glob",
    "-i", f"{output_dir}/frame*.png",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    video_path
]

try:
    result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    print(f"✓ Video created: {video_path}")
    print(f"  Duration: {num_frames/fps:.2f} seconds at {fps} fps")
except subprocess.CalledProcessError as e:
    print(f"Error creating video: {e}")
    print(f"STDERR: {e.stderr}")
except FileNotFoundError:
    print("Error: ffmpeg not found. Please install ffmpeg:")
    print("  sudo apt install ffmpeg")
