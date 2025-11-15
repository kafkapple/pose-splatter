"""
Comprehensive result analysis script for Pose Splatter experiments.
Analyzes metrics, generates visualizations, and creates summary reports.
"""
__date__ = "November 2025"

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

from src.config_utils import Config


def load_metrics(metrics_file):
    """Load metrics from CSV file."""
    if not Path(metrics_file).exists():
        print(f"Warning: Metrics file {metrics_file} not found")
        return None

    df = pd.read_csv(metrics_file, delimiter=',')
    return df


def plot_metrics_comparison(metrics_dict, output_dir):
    """Plot comparison of metrics across different views."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metrics Comparison Across Views', fontsize=16)

    metric_names = ['iou', 'l1', 'psnr', 'soft_iou', 'ssim']

    for idx, metric in enumerate(metric_names):
        ax = axes[idx // 3, idx % 3]

        for name, df in metrics_dict.items():
            if df is not None and metric in df.columns:
                ax.plot(df[metric], marker='o', label=name, linewidth=2)

        ax.set_xlabel('Camera View', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} per View', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    output_path = Path(output_dir) / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison plot: {output_path}")
    plt.close()


def plot_metric_heatmap(df, output_dir):
    """Plot heatmap of all metrics."""
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.T, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Value'})
    ax.set_xlabel('Camera View', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title('Metrics Heatmap', fontsize=14)

    plt.tight_layout()
    output_path = Path(output_dir) / 'metrics_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics heatmap: {output_path}")
    plt.close()


def generate_summary_report(config, metrics_dict, output_dir):
    """Generate text summary report."""
    report_path = Path(output_dir) / 'analysis_summary.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("POSE SPLATTER EXPERIMENT ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  - Data Directory: {config.data_directory}\n")
        f.write(f"  - Project Directory: {config.project_directory}\n")
        f.write(f"  - Image Size: {config.image_width}x{config.image_height}\n")
        f.write(f"  - Downsample: {config.image_downsample}x\n")
        f.write(f"  - Grid Size: {config.grid_size}\n")
        f.write(f"  - Learning Rate: {config.lr}\n")
        f.write(f"  - Holdout Views: {config.holdout_views}\n")
        f.write(f"  - FPS: {config.fps}\n\n")

        for split, df in metrics_dict.items():
            if df is None:
                continue

            f.write(f"\n{split.upper()} Metrics:\n")
            f.write("-" * 70 + "\n")

            # Calculate mean and std for each metric
            for col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                max_val = df[col].max()
                min_val = df[col].min()

                f.write(f"  {col.upper()}:\n")
                f.write(f"    Mean: {mean_val:.6f} ± {std_val:.6f}\n")
                f.write(f"    Range: [{min_val:.6f}, {max_val:.6f}]\n")
                f.write(f"    Per-view: {df[col].values}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Analysis completed successfully\n")
        f.write("=" * 70 + "\n")

    print(f"Saved summary report: {report_path}")
    return report_path


def compare_with_baseline(current_metrics, baseline_metrics):
    """Compare current metrics with baseline."""
    if baseline_metrics is None:
        return None

    improvements = {}
    for metric in current_metrics.columns:
        if metric in baseline_metrics.columns:
            current_mean = current_metrics[metric].mean()
            baseline_mean = baseline_metrics[metric].mean()

            # For loss metrics (l1), lower is better
            if metric in ['l1']:
                improvement = ((baseline_mean - current_mean) / baseline_mean) * 100
            else:  # For quality metrics (psnr, ssim, iou), higher is better
                improvement = ((current_mean - baseline_mean) / baseline_mean) * 100

            improvements[metric] = improvement

    return improvements


def main():
    parser = argparse.ArgumentParser(description="Analyze Pose Splatter results")
    parser.add_argument("config", type=str, help="Path to config JSON file")
    parser.add_argument("--baseline", type=str, default=None,
                       help="Path to baseline metrics CSV for comparison")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for analysis results")

    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Set output directory
    if args.output_dir is None:
        output_dir = Path(config.project_directory) / "analysis"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analysis output directory: {output_dir}")

    # Load metrics
    metrics_dict = {}
    for split in ['train', 'valid', 'test']:
        metrics_file = Path(config.project_directory) / f"metrics_{split}.csv"
        metrics_dict[split] = load_metrics(metrics_file)

    # Load baseline if provided
    baseline_metrics = None
    if args.baseline:
        baseline_metrics = load_metrics(args.baseline)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_metrics_comparison(metrics_dict, output_dir)

    for split, df in metrics_dict.items():
        if df is not None:
            plot_metric_heatmap(df, output_dir / split)

    # Generate summary report
    print("\nGenerating summary report...")
    report_path = generate_summary_report(config, metrics_dict, output_dir)

    # Compare with baseline
    if baseline_metrics is not None and metrics_dict['test'] is not None:
        print("\nComparing with baseline...")
        improvements = compare_with_baseline(metrics_dict['test'], baseline_metrics)

        print("\nImprovement over baseline:")
        for metric, improvement in improvements.items():
            direction = "↑" if improvement > 0 else "↓"
            print(f"  {metric.upper()}: {direction} {abs(improvement):.2f}%")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
