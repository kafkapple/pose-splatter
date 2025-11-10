"""
Interactive Gaussian visualization using Rerun.io.
Provides real-time 3D visualization with timeline support for animation sequences.

Installation:
    pip install rerun-sdk

Usage:
    # Save to .rrd file (for SSH/headless environments)
    python3 visualize_gaussian_rerun.py gaussians/gaussian_frame0000.npz --save

    # Animation sequence
    python3 visualize_gaussian_rerun.py animation/gaussians_npz/ --sequence --save

    # Web viewer (accessible via browser)
    python3 visualize_gaussian_rerun.py file.npz --web

    # GUI viewer (requires display)
    python3 visualize_gaussian_rerun.py file.npz
"""
import argparse
import numpy as np
import os
from pathlib import Path

try:
    import rerun as rr
except ImportError:
    print("Error: rerun-sdk not installed")
    print("Install with: pip install rerun-sdk")
    exit(1)


def visualize_single_frame(npz_file, point_size=0.005, save_path=None, web_port=None):
    """
    Visualize single Gaussian NPZ file with Rerun.

    Args:
        npz_file: Path to NPZ file
        point_size: Size of points in visualization
        save_path: If provided, save to .rrd file instead of opening viewer
        web_port: If provided, start web server on this port
    """
    # Load data
    data = np.load(npz_file)
    means = data['means']
    colors = data['colors']
    opacities = data['opacities']
    scales = data['scales']
    quaternions = data['quaternions']

    num_gaussians = len(means)

    print(f"========================================")
    print(f"Rerun Interactive Visualization")
    print(f"========================================")
    print(f"File: {npz_file}")
    print(f"Gaussians: {num_gaussians}")
    print(f"Point size: {point_size}")
    print("")

    # Initialize Rerun based on mode
    if save_path:
        # Save to .rrd file
        rr.init("Gaussian Splatting Viewer", recording_id="gaussian_viz")
        print(f"Mode: Save to file")
        print(f"Output: {save_path}")
    elif web_port:
        # Web viewer mode (Rerun 0.26.x)
        rr.init("Gaussian Splatting Viewer", spawn=False)
        rr.connect(f"127.0.0.1:{web_port}")
        print(f"Mode: Web viewer")
        print(f"Open browser: http://localhost:{web_port}")
        print(f"Note: Make sure to start 'rerun' web server first:")
        print(f"  rerun --port {web_port}")
    else:
        # Try GUI mode
        print("Mode: GUI viewer (attempting to spawn)")
        rr.init("Gaussian Splatting Viewer", spawn=True)

    # Log the point cloud with colors
    rr.log(
        "gaussian/points",
        rr.Points3D(
            positions=means,
            colors=(colors * 255).astype(np.uint8),
            radii=np.full(num_gaussians, point_size),
        )
    )

    # Log opacity as separate visualization
    opacity_colors = np.stack([opacities, opacities, opacities], axis=-1).squeeze()
    rr.log(
        "gaussian/opacity",
        rr.Points3D(
            positions=means,
            colors=(opacity_colors * 255).astype(np.uint8),
            radii=np.full(num_gaussians, point_size),
        )
    )

    # Log scale visualization
    avg_scale = scales.mean(axis=1, keepdims=True)
    scale_normalized = (avg_scale - avg_scale.min()) / (avg_scale.max() - avg_scale.min() + 1e-6)
    scale_colors = np.repeat(scale_normalized, 3, axis=1)
    rr.log(
        "gaussian/scale",
        rr.Points3D(
            positions=means,
            colors=(scale_colors * 255).astype(np.uint8),
            radii=avg_scale.squeeze() * 100,
        )
    )

    # Log statistics
    stats_text = f"""
# Gaussian Statistics

- **Total Gaussians**: {num_gaussians}
- **Opacity**: mean={opacities.mean():.4f}, std={opacities.std():.4f}
- **Scale X**: mean={scales[:, 0].mean():.4f}, std={scales[:, 0].std():.4f}
- **Scale Y**: mean={scales[:, 1].mean():.4f}, std={scales[:, 1].std():.4f}
- **Scale Z**: mean={scales[:, 2].mean():.4f}, std={scales[:, 2].std():.4f}
- **Color R**: mean={colors[:, 0].mean():.4f}, std={colors[:, 0].std():.4f}
- **Color G**: mean={colors[:, 1].mean():.4f}, std={colors[:, 1].std():.4f}
- **Color B**: mean={colors[:, 2].mean():.4f}, std={colors[:, 2].std():.4f}
"""
    rr.log("stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))

    # Save if requested
    if save_path:
        rr.save(save_path)
        print(f"\n✓ Saved to: {save_path}")
        print(f"View with: rerun {save_path}")
    elif web_port:
        print(f"\n✓ Web server running on port {web_port}")
        print("Press Ctrl+C to stop")
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping web server...")
    else:
        print("\n✓ Rerun viewer launched!")
        print("Controls:")
        print("  - Left click + drag: Rotate")
        print("  - Right click + drag: Pan")
        print("  - Scroll: Zoom")
        print("  - Toggle layers in left panel")


def visualize_animation_sequence(npz_dir, point_size=0.005, save_path=None, web_port=None):
    """
    Visualize animation sequence with timeline.
    """
    npz_files = sorted(Path(npz_dir).glob("*.npz"))

    if not npz_files:
        print(f"Error: No NPZ files found in {npz_dir}")
        return

    print(f"========================================")
    print(f"Rerun Animation Sequence Viewer")
    print(f"========================================")
    print(f"Directory: {npz_dir}")
    print(f"Frames: {len(npz_files)}")
    print(f"Point size: {point_size}")
    print("")

    # Initialize Rerun
    if save_path:
        rr.init("Gaussian Animation Viewer", recording_id="gaussian_anim")
        print(f"Mode: Save to file")
        print(f"Output: {save_path}")
    elif web_port:
        rr.init("Gaussian Animation Viewer", spawn=False)
        rr.connect(f"127.0.0.1:{web_port}")
        print(f"Mode: Web viewer")
        print(f"Open browser: http://localhost:{web_port}")
        print(f"Note: Make sure to start 'rerun' web server first:")
        print(f"  rerun --port {web_port}")
    else:
        print("Mode: GUI viewer (attempting to spawn)")
        rr.init("Gaussian Animation Viewer", spawn=True)

    # Log each frame
    for frame_idx, npz_file in enumerate(npz_files):
        rr.set_time_sequence("frame", frame_idx)

        data = np.load(npz_file)
        means = data['means']
        colors = data['colors']
        opacities = data['opacities']
        scales = data['scales']

        num_gaussians = len(means)

        # Log colored point cloud
        rr.log(
            "animation/points",
            rr.Points3D(
                positions=means,
                colors=(colors * 255).astype(np.uint8),
                radii=np.full(num_gaussians, point_size),
            )
        )

        # Log opacity
        opacity_colors = np.stack([opacities, opacities, opacities], axis=-1).squeeze()
        rr.log(
            "animation/opacity",
            rr.Points3D(
                positions=means,
                colors=(opacity_colors * 255).astype(np.uint8),
                radii=np.full(num_gaussians, point_size),
            )
        )

        # Log frame info
        frame_text = f"**Frame {frame_idx}**: {npz_file.name} ({num_gaussians} Gaussians)"
        rr.log("frame_info", rr.TextDocument(frame_text, media_type=rr.MediaType.MARKDOWN))

        print(f"  Logged frame {frame_idx}/{len(npz_files)-1}: {npz_file.name}")

    # Save or serve
    if save_path:
        rr.save(save_path)
        print(f"\n✓ Saved to: {save_path}")
        print(f"View with: rerun {save_path}")
    elif web_port:
        print(f"\n✓ Web server running on port {web_port}")
        print("Use timeline at bottom to scrub through frames")
        print("Press Ctrl+C to stop")
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping web server...")
    else:
        print("\n✓ Animation sequence loaded!")
        print("Use timeline at bottom to scrub through frames")


def visualize_comparison(npz_files, point_size=0.005, save_path=None, web_port=None):
    """
    Visualize multiple NPZ files side-by-side.
    """
    print(f"========================================")
    print(f"Rerun Comparison Viewer")
    print(f"========================================")
    print(f"Files: {len(npz_files)}")
    print("")

    # Initialize
    if save_path:
        rr.init("Gaussian Comparison Viewer", recording_id="gaussian_compare")
    elif web_port:
        rr.init("Gaussian Comparison Viewer", spawn=False)
        rr.connect(f"127.0.0.1:{web_port}")
        print(f"Web server: http://localhost:{web_port}")
        print(f"Note: Make sure to start 'rerun' web server first:")
        print(f"  rerun --port {web_port}")
    else:
        rr.init("Gaussian Comparison Viewer", spawn=True)

    for idx, npz_file in enumerate(npz_files):
        data = np.load(npz_file)
        means = data['means']
        colors = data['colors']

        offset = np.array([idx * 0.3, 0, 0])
        means_offset = means + offset

        rr.log(
            f"comparison/frame_{idx:02d}",
            rr.Points3D(
                positions=means_offset,
                colors=(colors * 255).astype(np.uint8),
                radii=np.full(len(means), point_size),
            )
        )

        print(f"  Logged: {Path(npz_file).name}")

    if save_path:
        rr.save(save_path)
        print(f"\n✓ Saved to: {save_path}")
    elif web_port:
        print("\n✓ Web server running. Press Ctrl+C to stop")
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive Gaussian visualization with Rerun.io")
    parser.add_argument("input", type=str, help="NPZ file or directory")
    parser.add_argument("--sequence", action="store_true", help="Animation sequence mode")
    parser.add_argument("--compare", nargs='+', help="Compare multiple files")
    parser.add_argument("--point_size", type=float, default=0.005, help="Point size")
    parser.add_argument("--save", type=str, help="Save to .rrd file instead of viewer")
    parser.add_argument("--web", type=int, default=None, const=9090, nargs='?',
                       help="Start web server on port (default: 9090)")

    args = parser.parse_args()

    # Determine save path
    save_path = args.save
    if save_path and not save_path.endswith('.rrd'):
        save_path += '.rrd'

    if args.compare:
        visualize_comparison(args.compare, args.point_size, save_path, args.web)
    elif args.sequence:
        visualize_animation_sequence(args.input, args.point_size, save_path, args.web)
    else:
        if os.path.isfile(args.input):
            visualize_single_frame(args.input, args.point_size, save_path, args.web)
        else:
            print(f"Error: {args.input} is not a file")
            print("Hint: Use --sequence for directories")
