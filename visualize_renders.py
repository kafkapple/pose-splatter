"""
Visualize and compare rendered images with ground truth.
"""
__date__ = "November 2025"

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_images_from_h5(h5_file, frame_idx, view_idx=None):
    """Load images from HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        if 'images' not in f:
            print(f"No 'images' dataset found in {h5_file}")
            return None

        images = f['images'][frame_idx]  # [C, h, w, channels]

        if view_idx is not None:
            images = images[view_idx]

    return images


def visualize_comparison(gt_images, pred_images, frame_idx, output_path):
    """Create comparison visualization."""
    num_views = gt_images.shape[0]
    fig, axes = plt.subplots(3, num_views, figsize=(3 * num_views, 9))

    if num_views == 1:
        axes = axes.reshape(-1, 1)

    for view in range(num_views):
        # Ground truth RGB
        gt_rgb = gt_images[view, :, :, :3]
        axes[0, view].imshow(gt_rgb)
        axes[0, view].set_title(f'GT View {view}')
        axes[0, view].axis('off')

        # Predicted RGB
        pred_rgb = pred_images[view, :, :, :3]
        axes[1, view].imshow(pred_rgb)
        axes[1, view].set_title(f'Pred View {view}')
        axes[1, view].axis('off')

        # Predicted Alpha
        pred_alpha = pred_images[view, :, :, 3]
        axes[2, view].imshow(pred_alpha, cmap='gray')
        axes[2, view].set_title(f'Alpha View {view}')
        axes[2, view].axis('off')

    fig.suptitle(f'Frame {frame_idx} - GT vs Predicted', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison: {output_path}")
    plt.close()


def create_video_grid(h5_file, frame_indices, output_dir):
    """Create grid of frames from video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_file, 'r') as f:
        images = f['images']
        num_frames = images.shape[0]
        num_views = images.shape[1]

        print(f"Total frames: {num_frames}, Views: {num_views}")

        for frame_idx in frame_indices:
            if frame_idx >= num_frames:
                print(f"Warning: Frame {frame_idx} out of range")
                continue

            frame = images[frame_idx]  # [C, h, w, channels]
            frame = frame.astype(np.float32) / 255.0

            fig, axes = plt.subplots(1, num_views, figsize=(4 * num_views, 4))

            if num_views == 1:
                axes = [axes]

            for view in range(num_views):
                if frame.shape[-1] == 4:
                    # RGBA
                    rgb = frame[view, :, :, :3]
                    alpha = frame[view, :, :, 3:4]
                    # Composite with white background
                    display = rgb * alpha + (1 - alpha)
                else:
                    # RGB only
                    display = frame[view, :, :, :3]

                axes[view].imshow(display)
                axes[view].set_title(f'View {view}')
                axes[view].axis('off')

            plt.tight_layout()
            output_path = output_dir / f'frame_{frame_idx:05d}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Saved {len(frame_indices)} frame grids to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize rendered images")
    parser.add_argument("--gt_file", type=str, default=None,
                       help="Ground truth HDF5 file")
    parser.add_argument("--pred_file", type=str, default=None,
                       help="Predicted/rendered HDF5 file")
    parser.add_argument("--frames", type=int, nargs='+', default=[0, 100, 500, 1000],
                       help="Frame indices to visualize")
    parser.add_argument("--output_dir", type=str, default="visualization",
                       help="Output directory")
    parser.add_argument("--mode", type=str, choices=['compare', 'grid'], default='compare',
                       help="Visualization mode")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'compare':
        if not args.gt_file or not args.pred_file:
            print("Error: Both --gt_file and --pred_file required for comparison mode")
            return

        print("Creating comparison visualizations...")
        for frame_idx in args.frames:
            gt_images = load_images_from_h5(args.gt_file, frame_idx)
            pred_images = load_images_from_h5(args.pred_file, frame_idx)

            if gt_images is None or pred_images is None:
                continue

            # Convert to float
            gt_images = gt_images.astype(np.float32) / 255.0
            pred_images = pred_images.astype(np.float32) / 255.0

            output_path = output_dir / f'comparison_frame_{frame_idx:05d}.png'
            visualize_comparison(gt_images, pred_images, frame_idx, output_path)

    elif args.mode == 'grid':
        if args.pred_file:
            print(f"Creating grid from: {args.pred_file}")
            create_video_grid(args.pred_file, args.frames, output_dir / 'pred_grids')

        if args.gt_file:
            print(f"Creating grid from: {args.gt_file}")
            create_video_grid(args.gt_file, args.frames, output_dir / 'gt_grids')

    print(f"\nVisualization complete! Results in: {output_dir}")


if __name__ == '__main__':
    main()
