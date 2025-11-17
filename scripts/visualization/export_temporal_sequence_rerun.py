"""
Export full temporal sequence and visualize with Rerun.

Renders all frames and creates both:
1. PNG sequence + MP4 video
2. Rerun .rrd file for interactive 3D visualization

Usage:
    # Export full sequence (3600 frames)
    python scripts/visualization/export_temporal_sequence_rerun.py \
        configs/baseline/markerless_mouse_nerf.json

    # Export subset
    python scripts/visualization/export_temporal_sequence_rerun.py \
        configs/baseline/markerless_mouse_nerf.json \
        --start 0 --end 1000 --view 0

    # Open in Rerun viewer
    rerun output/markerless_mouse_nerf/renders/full_sequence/sequence.rrd
"""
import argparse
import numpy as np
import os
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.utils import get_cam_params

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Warning: rerun-sdk not installed. Install with: pip install rerun-sdk")

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"


def export_sequence(config_path, start_frame=0, end_frame=None, view_num=0,
                    output_dir=None, save_images=True, save_rerun=True):
    """
    Export full temporal sequence with Rerun visualization.

    Args:
        config_path: Path to config JSON
        start_frame: Starting frame index
        end_frame: Ending frame index (None = all frames)
        view_num: Camera view number
        output_dir: Output directory
        save_images: Whether to save PNG images
        save_rerun: Whether to save Rerun .rrd file
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    config = Config(config_path)

    # Load camera parameters
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

    import h5py
    with h5py.File(img_fn, 'r') as f:
        total_frames = f['images'].shape[0]

    if end_frame is None:
        end_frame = total_frames

    print(f"Total frames in dataset: {total_frames}")
    print(f"Rendering frames: {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
    print(f"View: {view_num}")

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(config.project_directory, "renders", "full_sequence")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize model
    w = config.image_width // config.image_downsample
    h = config.image_height // config.image_downsample

    model = PoseSplatter(
        intrinsics=intrinsic,
        extrinsics=extrinsic,
        W=w,
        H=h,
        ell=config.ell,
        grid_size=config.grid_size,
        volume_idx=config.volume_idx,
        volume_fill_color=config.volume_fill_color,
        holdout_views=config.holdout_views,
        adaptive_camera=config.adaptive_camera,
        gaussian_mode=getattr(config, 'gaussian_mode', '3d'),
        gaussian_config=getattr(config, 'gaussian_config', {}),
    ).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(config.project_directory, config.model_fn)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize Rerun
    if save_rerun and RERUN_AVAILABLE:
        rr.init("pose_splatter_sequence", spawn=False)
        rrd_path = os.path.join(output_dir, "sequence.rrd")
        rr.save(rrd_path)
        print(f"Saving Rerun data to: {rrd_path}")

    # Load dataset for iteration
    dset = FrameDataset(img_fn, volume_fn, config.center_rotation_fn, C,
                        holdout_views=[], split="all")

    print("\nRendering frames...")
    with torch.no_grad():
        for frame_idx in tqdm(range(start_frame, min(end_frame, len(dset)))):
            batch = dset[frame_idx]

            # Move to device
            img = batch['img'].unsqueeze(0).to(device)
            mask = batch['mask'].unsqueeze(0).to(device)
            angle = batch['angle'].unsqueeze(0).to(device)
            p_3d = batch['p_3d'].unsqueeze(0).to(device)

            # Forward pass
            rgb_pred, _ = model(img, mask, p_3d, angle, view_num=view_num)

            # Save image
            if save_images:
                pred_img = rgb_pred[0].permute(1, 2, 0).cpu().numpy()
                pred_img = np.clip(pred_img * 255, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(pred_img)
                img_path = os.path.join(output_dir, f"frame{frame_idx:05d}.png")
                img_pil.save(img_path)

            # Log to Rerun
            if save_rerun and RERUN_AVAILABLE:
                rr.set_time_sequence("frame", frame_idx)

                # Log rendered image
                pred_img = rgb_pred[0].permute(1, 2, 0).cpu().numpy()
                rr.log("render/predicted", rr.Image(pred_img))

                # Get Gaussian parameters for 3D visualization
                if config.adaptive_camera:
                    volume, _ = model.carver(mask[:,None], img, p_3d, angle, adaptive=True)
                else:
                    volume = model.carver(mask[:,None], img, p_3d, angle, adaptive=False)

                volume = model.process_volume(volume[None])
                means, quats, scales, opacities, colors = model.get_gaussian_params_from_volume(volume)

                # Log 3D Gaussians
                means_np = means.cpu().numpy()
                colors_np = colors.cpu().numpy()
                opacities_np = opacities.cpu().numpy()

                # Filter by opacity threshold
                opacity_mask = opacities_np > 0.1
                filtered_means = means_np[opacity_mask]
                filtered_colors = colors_np[opacity_mask]
                filtered_opacities = opacities_np[opacity_mask]

                # Log as 3D points
                rr.log("gaussians/points",
                       rr.Points3D(
                           filtered_means,
                           colors=filtered_colors,
                           radii=filtered_opacities * 0.01
                       ))

    print(f"\n✓ Rendered {end_frame - start_frame} frames")

    if save_rerun and RERUN_AVAILABLE:
        print(f"✓ Saved Rerun data: {rrd_path}")
        print(f"\nTo view: rerun {rrd_path}")

    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export temporal sequence with Rerun")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame (default: all)")
    parser.add_argument("--view", type=int, default=0, help="Camera view number")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--no-images", action="store_true", help="Skip saving PNG images")
    parser.add_argument("--no-rerun", action="store_true", help="Skip Rerun export")

    args = parser.parse_args()

    export_sequence(
        args.config,
        start_frame=args.start,
        end_frame=args.end,
        view_num=args.view,
        output_dir=args.output,
        save_images=not args.no_images,
        save_rerun=not args.no_rerun
    )
