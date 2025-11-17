"""
Evaluate a trained Pose Splatter model.

Computes metrics (PSNR, SSIM, LPIPS) on test/validation views.
"""
__date__ = "November 2025"

import argparse
import json
import numpy as np
import os
import torch
from tqdm import tqdm

from src.config_utils import Config
from src.data import FrameDataset
from src.model import PoseSplatter
from src.utils import get_cam_params

# Import metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install lpips for perceptual metrics.")


def evaluate_model(config, checkpoint_path=None, output_json=None):
    """
    Evaluate model on holdout views.

    Args:
        config: Config object
        checkpoint_path: Path to checkpoint (default: config.model_fn)
        output_json: Path to save metrics JSON
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load camera parameters
    intrinsic, extrinsic, _ = get_cam_params(
        config.camera_fn,
        ds=config.image_downsample,
        up_fn=config.vertical_lines_fn,
        auto_orient=True,
        load_up_direction=not config.adaptive_camera,
    )
    C = len(intrinsic)
    print(f"Number of cameras: {C}")

    # Load dataset
    # Note: Config class automatically prepends project_directory to image_directory and volume_directory
    # Use .h5 format (FrameDataset will handle the correct path)
    img_fn = os.path.join(config.image_directory, "images.h5")
    volume_fn = os.path.join(config.volume_directory, "volumes.h5")

    print(f"Image dataset: {img_fn}")
    print(f"Volume dataset: {volume_fn}")

    # Create test dataset (holdout views only)
    test_dataset = FrameDataset(
        img_fn, volume_fn,
        config.center_rotation_fn,  # Config class automatically prepends project_directory
        C,
        holdout_views=config.holdout_views,
        split="test"
    )

    print(f"Test dataset size: {len(test_dataset)} frames")
    print(f"Holdout views: {config.holdout_views}")

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
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.project_directory, config.model_fn)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize metrics
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    if LPIPS_AVAILABLE:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

    # Evaluation loop
    all_psnr = []
    all_ssim = []
    all_lpips = [] if LPIPS_AVAILABLE else None

    print("\nEvaluating...")
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            batch = test_dataset[idx]

            # Move to device
            img = batch['img'].unsqueeze(0).to(device)
            mask = batch['mask'].unsqueeze(0).to(device)
            angle = batch['angle'].unsqueeze(0).to(device)
            p_3d = batch['p_3d'].unsqueeze(0).to(device)
            view_nums = batch['view_nums']

            # Forward pass through model
            rgb_pred, _ = model(
                img, mask, p_3d, angle,
                view_num=view_nums[0].item() if len(view_nums) > 0 else 0
            )

            # Get ground truth
            gt_img = img[0, view_nums[0]].permute(1, 2, 0)  # [H, W, 3]
            pred_img = rgb_pred[0].permute(1, 2, 0)  # [H, W, 3]

            # Ensure values in [0, 1]
            gt_img = torch.clamp(gt_img, 0, 1)
            pred_img = torch.clamp(pred_img, 0, 1)

            # Compute metrics (need [B, C, H, W] format)
            gt_batch = gt_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            pred_batch = pred_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

            psnr = psnr_metric(pred_batch, gt_batch)
            ssim = ssim_metric(pred_batch, gt_batch)

            all_psnr.append(psnr.item())
            all_ssim.append(ssim.item())

            if LPIPS_AVAILABLE:
                # LPIPS expects [-1, 1] range
                lpips = lpips_metric(pred_batch * 2 - 1, gt_batch * 2 - 1)
                all_lpips.append(lpips.item())

    # Compute statistics
    results = {
        "psnr_mean": float(np.mean(all_psnr)),
        "psnr_std": float(np.std(all_psnr)),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "num_frames": len(all_psnr),
        "holdout_views": config.holdout_views,
    }

    if LPIPS_AVAILABLE and all_lpips:
        results["lpips_mean"] = float(np.mean(all_lpips))
        results["lpips_std"] = float(np.std(all_lpips))

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"PSNR:  {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"SSIM:  {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    if LPIPS_AVAILABLE:
        print(f"LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
    print(f"Frames evaluated: {results['num_frames']}")
    print(f"Holdout views: {results['holdout_views']}")
    print("="*50)

    # Save results
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Pose Splatter model")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: from config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save metrics JSON")

    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Set default output path
    if args.output is None:
        args.output = os.path.join(config.project_directory, "evaluation_metrics.json")

    # Run evaluation
    evaluate_model(config, args.checkpoint, args.output)
