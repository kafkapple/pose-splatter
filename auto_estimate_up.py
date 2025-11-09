"""
Automatically estimate up direction from camera parameters (no GUI needed).
"""

import numpy as np
import h5py
import sys

def auto_estimate_up(camera_fn, output_fn):
    """
    Automatically estimate up direction from camera extrinsics.
    Uses the average y-axis direction of all cameras.
    """
    with h5py.File(camera_fn, 'r') as f:
        rotation = np.array(f['camera_parameters']['rotation'])  # [C, 3, 3]

    # The y-axis of each camera in world coordinates
    # R @ [0, 1, 0]^T gives the y-axis direction
    y_axes = rotation[:, :, 1]  # [C, 3]

    # Average y-axis
    up = y_axes.mean(axis=0)  # [3]
    up = up / np.linalg.norm(up)  # Normalize

    print(f"Estimated up direction: {up}")
    print(f"  Magnitude: {np.linalg.norm(up):.6f}")

    # Save
    np.savez(output_fn, up=up)
    print(f"Saved to: {output_fn}")

    # Verify
    data = np.load(output_fn)
    print(f"\nVerification:")
    print(f"  Keys: {list(data.keys())}")
    print(f"  up: {data['up']}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python auto_estimate_up.py <camera.h5> <output.npz>")
        sys.exit(1)

    auto_estimate_up(sys.argv[1], sys.argv[2])
