"""
Export multi-frame animation sequence.
Generates point clouds and Gaussian parameters for a sequence of frames.
"""
import argparse
import numpy as np
import os
import torch
from tqdm import tqdm

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.utils import get_cam_params

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"


def export_animation_sequence(config_path, start_frame, num_frames, output_dir,
                              export_ply=True, export_npz=True, export_json=False):
    """
    Export animation sequence with multiple frames.

    Args:
        config_path: Path to config JSON
        start_frame: Starting frame number
        num_frames: Number of frames to export
        output_dir: Output directory
        export_ply: Export PLY point clouds
        export_npz: Export NPZ Gaussian parameters
        export_json: Export JSON metadata
    """

    config = Config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"========================================")
    print(f"Animation Sequence Export")
    print(f"========================================")
    print(f"Device: {device}")
    print(f"Frames: {start_frame} to {start_frame + num_frames - 1} ({num_frames} total)")
    print(f"Output: {output_dir}")
    print("")

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

    # Create output directories
    if export_ply:
        ply_dir = os.path.join(output_dir, "pointclouds")
        os.makedirs(ply_dir, exist_ok=True)
    if export_npz:
        npz_dir = os.path.join(output_dir, "gaussians_npz")
        os.makedirs(npz_dir, exist_ok=True)
    if export_json:
        json_dir = os.path.join(output_dir, "gaussians_json")
        os.makedirs(json_dir, exist_ok=True)

    # Export each frame
    for frame_idx in tqdm(range(start_frame, start_frame + num_frames),
                          desc="Exporting frames"):

        # Get data for specified frame (view 0)
        dset_idx = C * frame_idx + 0

        try:
            mask, img, p_3d, angle, view_idx = dset.__getitem__(dset_idx)
        except:
            print(f"  ⚠ Frame {frame_idx} not available, skipping...")
            continue

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

        # Export in requested formats
        if export_ply:
            ply_fn = os.path.join(ply_dir, f"frame{frame_idx:04d}.ply")
            save_ply(means_np, colors_np, opacities_np, ply_fn)

        if export_npz:
            npz_fn = os.path.join(npz_dir, f"frame{frame_idx:04d}.npz")
            save_npz(means_np, quats_np, scales_np, opacities_np, colors_np, center, npz_fn)

        if export_json:
            import json
            json_fn = os.path.join(json_dir, f"frame{frame_idx:04d}.json")
            save_json_metadata(means_np, quats_np, scales_np, opacities_np,
                             colors_np, center, frame_idx, json_fn)

    print("")
    print("========================================")
    print("✓ Animation sequence export complete!")
    print("========================================")
    print(f"Exported {num_frames} frames")
    if export_ply:
        print(f"  PLY files: {ply_dir}")
    if export_npz:
        print(f"  NPZ files: {npz_dir}")
    if export_json:
        print(f"  JSON files: {json_dir}")
    print("")


def save_ply(positions, colors, opacities, filename):
    """Save point cloud as PLY file."""
    colors = np.clip(colors, 0, 1)
    colors_uint8 = (colors * 255).astype(np.uint8)
    opacities_uint8 = (opacities * 255).astype(np.uint8).reshape(-1)

    with open(filename, 'w') as f:
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

        for i in range(len(positions)):
            f.write(f"{positions[i,0]} {positions[i,1]} {positions[i,2]} ")
            f.write(f"{colors_uint8[i,0]} {colors_uint8[i,1]} {colors_uint8[i,2]} ")
            f.write(f"{opacities_uint8[i]}\n")


def save_npz(means, quats, scales, opacities, colors, center, filename):
    """Save all Gaussian parameters as NPZ."""
    np.savez_compressed(
        filename,
        means=means,
        quaternions=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        center=center
    )


def save_json_metadata(means, quats, scales, opacities, colors, center, frame_num, filename):
    """Save metadata as JSON."""
    import json

    data = {
        'frame': frame_num,
        'num_gaussians': len(means),
        'center': center.tolist(),
        'bounds': {
            'min': means.min(axis=0).tolist(),
            'max': means.max(axis=0).tolist()
        },
        'statistics': {
            'opacity_mean': float(opacities.mean()),
            'opacity_std': float(opacities.std()),
            'scale_mean': scales.mean(axis=0).tolist(),
            'scale_std': scales.std(axis=0).tolist()
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export animation sequence")
    parser.add_argument("--config", type=str, default="configs/markerless_mouse_nerf.json",
                       help="Path to config file")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Starting frame number")
    parser.add_argument("--num_frames", type=int, default=30,
                       help="Number of frames to export")
    parser.add_argument("--output_dir", type=str, default="output/markerless_mouse_nerf/animation",
                       help="Output directory")
    parser.add_argument("--ply", action="store_true", default=True,
                       help="Export PLY point clouds")
    parser.add_argument("--npz", action="store_true", default=True,
                       help="Export NPZ Gaussian parameters")
    parser.add_argument("--json", action="store_true", default=False,
                       help="Export JSON metadata")

    args = parser.parse_args()

    export_animation_sequence(
        args.config,
        args.start_frame,
        args.num_frames,
        args.output_dir,
        export_ply=args.ply,
        export_npz=args.npz,
        export_json=args.json
    )
