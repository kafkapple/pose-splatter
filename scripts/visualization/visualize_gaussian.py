"""
Simple Gaussian NPZ visualization script.
Visualizes Gaussian parameters using matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def visualize_gaussian_npz(npz_file, max_points=5000):
    """
    Visualize Gaussian parameters from NPZ file.

    Args:
        npz_file: Path to NPZ file
        max_points: Maximum number of points to display
    """

    # Load data
    data = np.load(npz_file)
    means = data['means']
    colors = data['colors']
    opacities = data['opacities']
    scales = data['scales']
    quaternions = data['quaternions']

    print(f"========================================")
    print(f"Gaussian NPZ Visualization")
    print(f"========================================")
    print(f"File: {npz_file}")
    print(f"Total Gaussians: {len(means)}")
    print(f"Displaying: {min(max_points, len(means))} points")
    print("")

    # Sample points if too many
    if len(means) > max_points:
        indices = np.random.choice(len(means), max_points, replace=False)
        means = means[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        scales = scales[indices]

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))

    # 1. 3D Point Cloud with colors
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(means[:, 0], means[:, 1], means[:, 2],
               c=colors, s=1, alpha=0.6)
    ax1.set_title('3D Point Cloud (Colored)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 2. Point Cloud colored by opacity
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter = ax2.scatter(means[:, 0], means[:, 1], means[:, 2],
                         c=opacities.squeeze(), s=1, cmap='viridis', alpha=0.6)
    ax2.set_title('Point Cloud (Colored by Opacity)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(scatter, ax=ax2, label='Opacity')

    # 3. Point Cloud colored by scale (average)
    avg_scale = scales.mean(axis=1)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    scatter = ax3.scatter(means[:, 0], means[:, 1], means[:, 2],
                         c=avg_scale, s=1, cmap='plasma', alpha=0.6)
    ax3.set_title('Point Cloud (Colored by Scale)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.colorbar(scatter, ax=ax3, label='Avg Scale')

    # 4. Opacity histogram
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(opacities.flatten(), bins=50, color='blue', alpha=0.7)
    ax4.set_title('Opacity Distribution')
    ax4.set_xlabel('Opacity')
    ax4.set_ylabel('Count')
    ax4.axvline(opacities.mean(), color='red', linestyle='--',
                label=f'Mean: {opacities.mean():.3f}')
    ax4.legend()

    # 5. Scale distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(scales[:, 0].flatten(), bins=50, alpha=0.5, label='Scale X', color='red')
    ax5.hist(scales[:, 1].flatten(), bins=50, alpha=0.5, label='Scale Y', color='green')
    ax5.hist(scales[:, 2].flatten(), bins=50, alpha=0.5, label='Scale Z', color='blue')
    ax5.set_title('Scale Distribution')
    ax5.set_xlabel('Scale')
    ax5.set_ylabel('Count')
    ax5.legend()

    # 6. Color distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(colors[:, 0].flatten(), bins=30, alpha=0.5, label='Red', color='red')
    ax6.hist(colors[:, 1].flatten(), bins=30, alpha=0.5, label='Green', color='green')
    ax6.hist(colors[:, 2].flatten(), bins=30, alpha=0.5, label='Blue', color='blue')
    ax6.set_title('Color Distribution')
    ax6.set_xlabel('Color Value')
    ax6.set_ylabel('Count')
    ax6.legend()

    plt.tight_layout()

    # Save figure
    output_file = npz_file.replace('.npz', '_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")

    # Show interactive plot
    plt.show()

    # Print statistics
    print("")
    print("========================================")
    print("Statistics")
    print("========================================")
    print(f"Opacity - Mean: {opacities.mean():.4f}, Std: {opacities.std():.4f}")
    print(f"Scale X - Mean: {scales[:, 0].mean():.4f}, Std: {scales[:, 0].std():.4f}")
    print(f"Scale Y - Mean: {scales[:, 1].mean():.4f}, Std: {scales[:, 1].std():.4f}")
    print(f"Scale Z - Mean: {scales[:, 2].mean():.4f}, Std: {scales[:, 2].std():.4f}")
    print(f"Color R - Mean: {colors[:, 0].mean():.4f}, Std: {colors[:, 0].std():.4f}")
    print(f"Color G - Mean: {colors[:, 1].mean():.4f}, Std: {colors[:, 1].std():.4f}")
    print(f"Color B - Mean: {colors[:, 2].mean():.4f}, Std: {colors[:, 2].std():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Gaussian NPZ file")
    parser.add_argument("npz_file", type=str, help="Path to NPZ file")
    parser.add_argument("--max_points", type=int, default=5000,
                       help="Maximum points to display (default: 5000)")

    args = parser.parse_args()

    visualize_gaussian_npz(args.npz_file, args.max_points)
