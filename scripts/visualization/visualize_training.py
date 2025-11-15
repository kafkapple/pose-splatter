"""
Visualize training progress and results.
Parse log files and create visualization plots.
"""
__date__ = "November 2025"

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_training_log(log_file):
    """Parse training log file to extract metrics."""
    metrics = {
        'epoch': [],
        'train_loss': [],
        'valid_loss': [],
        'train_psnr': [],
        'valid_psnr': [],
    }

    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return None

    with open(log_file, 'r') as f:
        for line in f:
            # Example pattern: "Epoch 1/50 - Train Loss: 0.1234, Valid Loss: 0.5678"
            epoch_match = re.search(r'Epoch (\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                metrics['epoch'].append(epoch)

            train_loss_match = re.search(r'Train Loss: ([\d.]+)', line)
            if train_loss_match:
                metrics['train_loss'].append(float(train_loss_match.group(1)))

            valid_loss_match = re.search(r'Valid Loss: ([\d.]+)', line)
            if valid_loss_match:
                metrics['valid_loss'].append(float(valid_loss_match.group(1)))

            train_psnr_match = re.search(r'Train PSNR: ([\d.]+)', line)
            if train_psnr_match:
                metrics['train_psnr'].append(float(train_psnr_match.group(1)))

            valid_psnr_match = re.search(r'Valid PSNR: ([\d.]+)', line)
            if valid_psnr_match:
                metrics['valid_psnr'].append(float(valid_psnr_match.group(1)))

    return metrics


def plot_training_curves(metrics, output_dir):
    """Plot training and validation curves."""
    if not metrics or not metrics['epoch']:
        print("No metrics to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    if metrics['train_loss']:
        ax1.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if metrics['valid_loss']:
        ax1.plot(metrics['epoch'], metrics['valid_loss'], 'r-', label='Valid Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PSNR curves
    if metrics['train_psnr']:
        ax2.plot(metrics['epoch'], metrics['train_psnr'], 'b-', label='Train PSNR', linewidth=2)
    if metrics['valid_psnr']:
        ax2.plot(metrics['epoch'], metrics['valid_psnr'], 'r-', label='Valid PSNR', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Training and Validation PSNR', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves: {output_path}")
    plt.close()


def plot_pipeline_timeline(log_dir, output_dir):
    """Parse step logs and visualize pipeline timeline."""
    steps = []
    step_files = sorted(Path(log_dir).glob('step*_*.log'))

    for step_file in step_files:
        step_name = step_file.stem
        # Try to extract timing info from log
        # This is a simple placeholder - adjust based on actual log format
        steps.append(step_name)

    if not steps:
        print("No step logs found")
        return

    print(f"Found {len(steps)} pipeline steps:")
    for step in steps:
        print(f"  - {step}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Training log file to parse")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Directory containing pipeline logs")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory for plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file:
        print(f"Parsing training log: {args.log_file}")
        metrics = parse_training_log(args.log_file)
        if metrics:
            plot_training_curves(metrics, output_dir)

    if args.log_dir:
        print(f"Analyzing pipeline logs in: {args.log_dir}")
        plot_pipeline_timeline(args.log_dir, output_dir)

    print("\nVisualization complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
