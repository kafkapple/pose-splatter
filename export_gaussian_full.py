"""
Export complete 3D Gaussian Splatting parameters for external viewers.
Exports all Gaussian parameters including means, quaternions, scales, opacities, and colors.
Compatible with Blender Gaussian Splatting plugins and specialized viewers.
"""
import argparse
import numpy as np
import os
import torch
import json

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.utils import get_cam_params

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"


def export_gaussian_full(config_path, frame_num, output_dir, format_type='npz'):
    """
    Export complete Gaussian parameters.

    Args:
        config_path: Path to config JSON
        frame_num: Frame number to export
        output_dir: Output directory
        format_type: 'npz', 'ply_extended', or 'json'
    """

    config = Config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Exporting full Gaussian parameters for frame {frame_num}")
    print(f"Device: {device}")
    print(f"Format: {format_type}")

    # Get camera parameters
    intrinsic, extrinsic, _ = get_cam_params(
        config.camera_fn,
        ds=config.image_downsample,
        up_fn=config.vertical_lines_fn,
        auto_orient=True,
        load_up_direction=not config.adaptive_camera,
    )
    C = len(intrinsic)

    # Load dataset
    img_fn = os.path.join(config.image_directory, "images.h5")
    volume_fn = os.path.join(config.volume_directory, "volumes.h5")
    dset = FrameDataset(
        img_fn,
        volume_fn,
        config.center_rotation_fn,
        C,
        holdout_views=config.holdout_views,
        split="all"
    )

    w = config.image_width // config.image_downsample
    h = config.image_height // config.image_downsample

    # Create model
    model = PoseSplatter(
        intrinsics=intrinsic,
        extrinsics=extrinsic,
        W=w,
        H=h,
        ell=config.ell,
        grid_size=config.grid_size,
        volume_idx=config.volume_idx,
        ablation=False,
        volume_fill_color=config.volume_fill_color,
        holdout_views=config.holdout_views,
        adaptive_camera=config.adaptive_camera
    )
    model.to(device)

    # Load trained weights
    model_fn = config.model_fn
    model.load_state_dict(torch.load(model_fn)["model_state_dict"])
    model.eval()

    # Get data for specified frame (view 0)
    dset_idx = C * frame_num + 0
    mask, img, p_3d, angle, view_idx = dset.__getitem__(dset_idx)

    print(f"Generating 3D volume and extracting Gaussians...")

    # Forward pass to generate volume and Gaussians
    with torch.no_grad():
        # Make the volume
        if config.adaptive_camera:
            volume, temp_K = model.carver(
                mask[:,None].to(device),
                img.to(device),
                p_3d.to(device),
                angle,
                adaptive=config.adaptive_camera
            )
        else:
            volume = model.carver(
                mask[:,None].to(device),
                img.to(device),
                p_3d.to(device),
                angle,
                adaptive=config.adaptive_camera
            )

        # Process volume through U-Nets
        volume = model.process_volume(volume[None])

        # Get ALL Gaussian parameters
        means, quats, scales, opacities, colors = model.get_gaussian_params_from_volume(volume)

        # Transform Gaussians to world coordinates
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = torch.tensor([[c,-s,0], [s,c,0], [0,0,1]]).to(volume.device, torch.float32)
        means = means @ rot_mat.T + p_3d.to(device, torch.float32)

        # Convert to numpy
        means_np = means.detach().cpu().numpy()
        quats_np = quats.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        opacities_np = opacities.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()

        # Center the point cloud
        center = np.mean(means_np, axis=0, keepdims=True)
        means_np -= center

        print(f"\nGaussian statistics:")
        print(f"  Number of Gaussians: {len(means_np)}")
        print(f"  Means range: {means_np.min(axis=0)} to {means_np.max(axis=0)}")
        print(f"  Quaternions range: {quats_np.min(axis=0)} to {quats_np.max(axis=0)}")
        print(f"  Scales range: {scales_np.min(axis=0)} to {scales_np.max(axis=0)}")
        print(f"  Opacities range: {opacities_np.min()} to {opacities_np.max()}")
        print(f"  Colors range: {colors_np.min()} to {colors_np.max()}")

    # Save in requested format
    os.makedirs(output_dir, exist_ok=True)

    if format_type == 'npz':
        output_fn = os.path.join(output_dir, f"gaussian_frame{frame_num:04d}.npz")
        save_npz(means_np, quats_np, scales_np, opacities_np, colors_np, center, output_fn)

    elif format_type == 'ply_extended':
        output_fn = os.path.join(output_dir, f"gaussian_frame{frame_num:04d}_extended.ply")
        save_ply_extended(means_np, quats_np, scales_np, opacities_np, colors_np, output_fn)

    elif format_type == 'json':
        output_fn = os.path.join(output_dir, f"gaussian_frame{frame_num:04d}.json")
        save_json(means_np, quats_np, scales_np, opacities_np, colors_np, center, output_fn)

    else:
        raise ValueError(f"Unknown format: {format_type}")

    print(f"\n✓ Export complete!")
    print(f"  File: {output_fn}")
    print(f"  Gaussians: {len(means_np)}")


def save_npz(means, quats, scales, opacities, colors, center, filename):
    """Save all Gaussian parameters as NPZ (NumPy archive)."""
    np.savez_compressed(
        filename,
        means=means,
        quaternions=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        center=center,
        metadata={
            'format': 'gaussian_splatting_full',
            'num_gaussians': len(means),
            'version': '1.0'
        }
    )
    print(f"Saved NPZ: {filename}")


def save_ply_extended(means, quats, scales, opacities, colors, filename):
    """
    Save extended PLY format with all Gaussian parameters.
    Compatible with advanced Gaussian Splatting viewers.
    """

    # Ensure colors are in [0, 1] range
    colors = np.clip(colors, 0, 1)
    colors_uint8 = (colors * 255).astype(np.uint8)
    opacities_uint8 = (opacities * 255).astype(np.uint8).reshape(-1)

    # Scale values for PLY storage
    scales_scaled = (scales * 1000).astype(np.int32)  # Store as millimeters
    quats_scaled = (quats * 32767).astype(np.int16)   # Normalize to int16 range

    with open(filename, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Gaussian Splatting Extended Format\n")
        f.write(f"element vertex {len(means)}\n")

        # Position
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        # Color
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")

        # Quaternion (rotation)
        f.write("property short quat_w\n")
        f.write("property short quat_x\n")
        f.write("property short quat_y\n")
        f.write("property short quat_z\n")

        # Scale
        f.write("property int scale_x\n")
        f.write("property int scale_y\n")
        f.write("property int scale_z\n")

        f.write("end_header\n")

        # Data
        for i in range(len(means)):
            # Position
            f.write(f"{means[i,0]} {means[i,1]} {means[i,2]} ")
            # Color
            f.write(f"{colors_uint8[i,0]} {colors_uint8[i,1]} {colors_uint8[i,2]} {opacities_uint8[i]} ")
            # Quaternion
            f.write(f"{quats_scaled[i,0]} {quats_scaled[i,1]} {quats_scaled[i,2]} {quats_scaled[i,3]} ")
            # Scale
            f.write(f"{scales_scaled[i,0]} {scales_scaled[i,1]} {scales_scaled[i,2]}\n")

    print(f"Saved extended PLY: {filename}")


def save_json(means, quats, scales, opacities, colors, center, filename):
    """Save as JSON for easy inspection and custom processing."""

    data = {
        'metadata': {
            'format': 'gaussian_splatting_full',
            'num_gaussians': len(means),
            'version': '1.0'
        },
        'center': center.tolist(),
        'gaussians': []
    }

    # Sample only first 100 gaussians for JSON (full data would be huge)
    num_sample = min(100, len(means))
    indices = np.linspace(0, len(means)-1, num_sample, dtype=int)

    for i in indices:
        data['gaussians'].append({
            'position': means[i].tolist(),
            'quaternion': quats[i].tolist(),
            'scale': scales[i].tolist(),
            'opacity': float(opacities[i]),
            'color': colors[i].tolist()
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON (sampled {num_sample} Gaussians): {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export full Gaussian Splatting parameters")
    parser.add_argument("--config", type=str, default="configs/markerless_mouse_nerf.json",
                       help="Path to config file")
    parser.add_argument("--frame", type=int, default=0,
                       help="Frame number to export")
    parser.add_argument("--output_dir", type=str, default="output/markerless_mouse_nerf/gaussians",
                       help="Output directory")
    parser.add_argument("--format", type=str, default="npz",
                       choices=['npz', 'ply_extended', 'json'],
                       help="Output format: npz (recommended), ply_extended, or json")

    args = parser.parse_args()

    export_gaussian_full(args.config, args.frame, args.output_dir, args.format)
