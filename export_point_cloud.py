"""
Export 3D Gaussian Splatting point cloud to PLY format.
"""
import argparse
import numpy as np
import os
import torch

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.utils import get_cam_params

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"


def export_point_cloud(config_path, frame_num, output_fn):
    """Export 3D point cloud with colors from the trained model."""

    config = Config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Exporting point cloud for frame {frame_num}")
    print(f"Device: {device}")

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

    print(f"Generating 3D volume...")

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

        # Get Gaussian parameters
        means, quats, scales, opacities, colors = model.get_gaussian_params_from_volume(volume)

        # Transform Gaussians to world coordinates
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = torch.tensor([[c,-s,0], [s,c,0], [0,0,1]]).to(volume.device, torch.float32)
        means = means @ rot_mat.T + p_3d.to(device, torch.float32)

        # Convert to numpy
        pos = means.detach().cpu().numpy()
        rgb = colors.detach().cpu().numpy()
        opacity = opacities.detach().cpu().numpy()

        # Center the point cloud
        pos -= np.mean(pos, axis=0, keepdims=True)

        print(f"Point cloud statistics:")
        print(f"  Number of points: {len(pos)}")
        print(f"  Position range: {pos.min(axis=0)} to {pos.max(axis=0)}")
        print(f"  RGB range: {rgb.min()} to {rgb.max()}")
        print(f"  Opacity range: {opacity.min()} to {opacity.max()}")

    # Save as PLY
    print(f"Saving to {output_fn}...")
    save_ply(pos, rgb, opacity, output_fn)

    print(f"✓ Point cloud exported successfully!")
    print(f"  File: {output_fn}")
    print(f"  Points: {len(pos)}")


def save_ply(positions, colors, opacities, filename):
    """Save point cloud as PLY file."""

    # Ensure colors are in [0, 1] range
    colors = np.clip(colors, 0, 1)

    # Convert to uint8 for PLY
    colors_uint8 = (colors * 255).astype(np.uint8)
    opacities_uint8 = (opacities * 255).astype(np.uint8).reshape(-1)

    # Write PLY file
    with open(filename, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(positions)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")

        # Data
        for i in range(len(positions)):
            f.write(f"{positions[i,0]} {positions[i,1]} {positions[i,2]} ")
            f.write(f"{colors_uint8[i,0]} {colors_uint8[i,1]} {colors_uint8[i,2]} ")
            f.write(f"{opacities_uint8[i]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export 3D point cloud to PLY")
    parser.add_argument("--config", type=str, default="configs/markerless_mouse_nerf.json",
                       help="Path to config file")
    parser.add_argument("--frame", type=int, default=0,
                       help="Frame number to export")
    parser.add_argument("--output", type=str, default=None,
                       help="Output PLY filename")

    args = parser.parse_args()

    if args.output is None:
        output_dir = "output/markerless_mouse_nerf/pointclouds"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"frame{args.frame:04d}.ply")

    export_point_cloud(args.config, args.frame, args.output)
