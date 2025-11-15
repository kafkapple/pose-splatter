#!/usr/bin/env python3
"""
2D vs 3D Gaussian Splatting 실험 결과 분석 스크립트

Usage:
    python3 scripts/analyze_results.py --log2d output/2d_short.log --log3d output/3d_short.log
"""

import argparse
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def parse_log_file(log_path: str) -> Dict:
    """
    학습 로그 파일을 파싱하여 metrics 추출

    Returns:
        {
            'losses': [float],  # Epoch별 total loss
            'iou': [float],
            'ssim': [float],
            'img': [float],
            'epochs': [int],
            'time_per_epoch': float,
            'final_metrics': dict
        }
    """
    with open(log_path, 'r') as f:
        content = f.read()

    # Epoch loss 추출
    loss_pattern = r'epoch loss: ([\d.]+)'
    losses = [float(m) for m in re.findall(loss_pattern, content)]

    # Epoch number 추출
    epoch_pattern = r'b (\d+):'
    epochs = [int(m) for m in re.findall(epoch_pattern, content)]

    # 개별 loss 추출 (더 복잡한 로그 형식 지원 필요 시)
    iou_losses = []
    ssim_losses = []
    img_losses = []

    # Validation metrics (있다면)
    val_pattern = r'validation.*iou: ([\d.]+).*psnr: ([\d.]+)'
    val_matches = re.findall(val_pattern, content)

    # Time per epoch 추정 (로그에 시간 정보가 있다면)
    # 간단히 총 epoch 수와 로그 크기로 추정

    results = {
        'losses': losses,
        'iou': iou_losses if iou_losses else losses,  # Fallback
        'ssim': ssim_losses,
        'img': img_losses,
        'epochs': epochs if epochs else list(range(len(losses))),
        'final_loss': losses[-1] if losses else None,
        'validation': val_matches,
    }

    return results


def compare_convergence(data_2d: Dict, data_3d: Dict) -> plt.Figure:
    """
    2D vs 3D Loss convergence 비교 그래프
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(data_2d['epochs'], data_2d['losses'], label='2D Mode', color='blue', linewidth=2)
    ax.plot(data_3d['epochs'], data_3d['losses'], label='3D Mode', color='red', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Loss reduction
    ax = axes[1]
    if len(data_2d['losses']) > 0 and len(data_3d['losses']) > 0:
        loss_reduction_2d = [(data_2d['losses'][0] - l) / data_2d['losses'][0] * 100
                             for l in data_2d['losses']]
        loss_reduction_3d = [(data_3d['losses'][0] - l) / data_3d['losses'][0] * 100
                             for l in data_3d['losses']]

        ax.plot(data_2d['epochs'], loss_reduction_2d, label='2D Mode', color='blue', linewidth=2)
        ax.plot(data_3d['epochs'], loss_reduction_3d, label='3D Mode', color='red', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Reduction (%)', fontsize=12)
        ax.set_title('Convergence Speed', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compare_metrics(data_2d: Dict, data_3d: Dict) -> plt.Figure:
    """
    Final metrics 비교 bar chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = {
        'Final Loss': [data_2d['final_loss'], data_3d['final_loss']],
    }

    # Validation metrics (있다면)
    if data_2d['validation'] and data_3d['validation']:
        val_2d = data_2d['validation'][-1]  # Last validation
        val_3d = data_3d['validation'][-1]
        metrics['Val IoU'] = [float(val_2d[0]), float(val_3d[0])]
        metrics['Val PSNR'] = [float(val_2d[1]), float(val_3d[1])]

    x = np.arange(len(metrics))
    width = 0.35

    metric_names = list(metrics.keys())
    values_2d = [metrics[m][0] for m in metric_names]
    values_3d = [metrics[m][1] for m in metric_names]

    bars1 = ax.bar(x - width/2, values_2d, width, label='2D Mode', color='blue', alpha=0.8)
    bars2 = ax.bar(x + width/2, values_3d, width, label='3D Mode', color='red', alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def generate_comparison_table(data_2d: Dict, data_3d: Dict) -> str:
    """
    Markdown 형식 비교 테이블 생성
    """
    table = "# 2D vs 3D Gaussian Splatting 비교 결과\n\n"
    table += "## Quantitative Comparison\n\n"
    table += "| Metric | 2D Mode | 3D Mode | Difference | Winner |\n"
    table += "|--------|---------|---------|------------|--------|\n"

    # Final Loss
    final_loss_2d = data_2d['final_loss']
    final_loss_3d = data_3d['final_loss']
    if final_loss_2d and final_loss_3d:
        diff = ((final_loss_2d - final_loss_3d) / final_loss_3d) * 100
        winner = "2D" if final_loss_2d < final_loss_3d else "3D"
        table += f"| Final Loss | {final_loss_2d:.6f} | {final_loss_3d:.6f} | {diff:+.2f}% | {winner} |\n"

    # Epochs
    epochs_2d = len(data_2d['losses'])
    epochs_3d = len(data_3d['losses'])
    table += f"| Epochs Trained | {epochs_2d} | {epochs_3d} | - | - |\n"

    # Params per Gaussian
    table += f"| Params/Gaussian | 9 | 14 | -35.7% | 2D |\n"

    # Validation (있다면)
    if data_2d['validation'] and data_3d['validation']:
        val_2d = data_2d['validation'][-1]
        val_3d = data_3d['validation'][-1]
        iou_2d, psnr_2d = float(val_2d[0]), float(val_2d[1])
        iou_3d, psnr_3d = float(val_3d[0]), float(val_3d[1])

        iou_diff = ((iou_2d - iou_3d) / iou_3d) * 100
        psnr_diff = ((psnr_2d - psnr_3d) / psnr_3d) * 100

        table += f"| Val IoU | {iou_2d:.4f} | {iou_3d:.4f} | {iou_diff:+.2f}% | {'2D' if iou_2d > iou_3d else '3D'} |\n"
        table += f"| Val PSNR | {psnr_2d:.2f} dB | {psnr_3d:.2f} dB | {psnr_diff:+.2f}% | {'2D' if psnr_2d > psnr_3d else '3D'} |\n"

    table += "\n## Analysis\n\n"

    # Convergence analysis
    if final_loss_2d and final_loss_3d:
        if final_loss_2d < final_loss_3d:
            table += "- **2D Mode** achieved lower final loss\n"
        else:
            table += "- **3D Mode** achieved lower final loss\n"

    # Memory efficiency
    table += "- **2D Mode** uses 35.7% fewer parameters per Gaussian (9 vs 14)\n"

    # Speed (추정)
    table += "- Training speed comparison requires timing data\n"

    table += "\n## Recommendation\n\n"
    table += "_To be filled after analyzing results_\n\n"

    return table


def main():
    parser = argparse.ArgumentParser(description='Analyze 2D vs 3D experiment results')
    parser.add_argument('--log2d', type=str, required=True,
                       help='Path to 2D training log file')
    parser.add_argument('--log3d', type=str, required=True,
                       help='Path to 3D training log file')
    parser.add_argument('--output', type=str, default='output/comparison_analysis',
                       help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("2D vs 3D Gaussian Splatting 실험 결과 분석")
    print("=" * 60)
    print()

    # Parse logs
    print(f"Parsing 2D log: {args.log2d}")
    data_2d = parse_log_file(args.log2d)
    print(f"  ✓ Found {len(data_2d['losses'])} epochs")
    print(f"  ✓ Final loss: {data_2d['final_loss']:.6f}" if data_2d['final_loss'] else "  ✗ No loss data")
    print()

    print(f"Parsing 3D log: {args.log3d}")
    data_3d = parse_log_file(args.log3d)
    print(f"  ✓ Found {len(data_3d['losses'])} epochs")
    print(f"  ✓ Final loss: {data_3d['final_loss']:.6f}" if data_3d['final_loss'] else "  ✗ No loss data")
    print()

    # Generate plots
    print("Generating comparison plots...")

    # Loss convergence
    fig1 = compare_convergence(data_2d, data_3d)
    plot1_path = output_dir / 'loss_convergence.png'
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot1_path}")

    # Metrics comparison
    fig2 = compare_metrics(data_2d, data_3d)
    plot2_path = output_dir / 'metrics_comparison.png'
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot2_path}")
    print()

    # Generate comparison table
    print("Generating comparison table...")
    table = generate_comparison_table(data_2d, data_3d)
    table_path = output_dir / 'comparison_table.md'
    with open(table_path, 'w') as f:
        f.write(table)
    print(f"  ✓ Saved: {table_path}")
    print()

    # Save raw data
    print("Saving raw data...")
    raw_data = {
        '2d': {k: v for k, v in data_2d.items() if k != 'validation'},
        '3d': {k: v for k, v in data_3d.items() if k != 'validation'},
    }
    data_path = output_dir / 'raw_data.json'
    with open(data_path, 'w') as f:
        json.dump(raw_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"  ✓ Saved: {data_path}")
    print()

    # Print summary
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print()
    print("Results:")
    print(f"  - Loss convergence plot: {plot1_path}")
    print(f"  - Metrics comparison: {plot2_path}")
    print(f"  - Comparison table: {table_path}")
    print(f"  - Raw data: {data_path}")
    print()

    # Print quick summary
    print("Quick Summary:")
    if data_2d['final_loss'] and data_3d['final_loss']:
        print(f"  2D Final Loss: {data_2d['final_loss']:.6f}")
        print(f"  3D Final Loss: {data_3d['final_loss']:.6f}")
        if data_2d['final_loss'] < data_3d['final_loss']:
            improvement = ((data_3d['final_loss'] - data_2d['final_loss']) / data_3d['final_loss']) * 100
            print(f"  → 2D is {improvement:.2f}% better")
        else:
            improvement = ((data_2d['final_loss'] - data_3d['final_loss']) / data_2d['final_loss']) * 100
            print(f"  → 3D is {improvement:.2f}% better")
    print()

    print("Next steps:")
    print("  1. Review plots and comparison table")
    print("  2. Check rendered images for qualitative comparison")
    print("  3. Update experiment report with findings")
    print()


if __name__ == '__main__':
    main()
