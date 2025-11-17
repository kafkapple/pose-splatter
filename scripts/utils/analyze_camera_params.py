#!/usr/bin/env python3
"""
Analyze new_cam.pkl structure and visualize camera parameters

This script loads the camera calibration file and provides detailed analysis
of each component with explanations.
"""

import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_camera_params(pkl_path):
    """
    Comprehensive analysis of camera parameters
    """
    print("=" * 80)
    print("CAMERA PARAMETERS ANALYSIS")
    print("=" * 80)
    print()

    # Load pickle file
    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        cameras = pickle.load(f)

    print(f"Number of cameras: {len(cameras)}")
    print(f"Data type: {type(cameras)}")
    print()

    # Analyze first camera in detail
    print("=" * 80)
    print("CAMERA 0 - DETAILED STRUCTURE")
    print("=" * 80)
    print()

    cam0 = cameras[0]
    print(f"Keys in camera dict: {list(cam0.keys())}")
    print()

    # 1. Intrinsic Matrix (K)
    print("-" * 80)
    print("1. INTRINSIC MATRIX (K) - Camera Internal Parameters")
    print("-" * 80)
    K = cam0['K']
    print(f"Shape: {K.shape}")
    print(f"Data type: {K.dtype}")
    print()
    print("Matrix:")
    print(K)
    print()

    # Extract and explain parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    print("Components:")
    print(f"  fx (focal length X): {fx:.2f} pixels")
    print(f"  fy (focal length Y): {fy:.2f} pixels")
    print(f"  cx (principal point X): {cx:.2f} pixels")
    print(f"  cy (principal point Y): {cy:.2f} pixels")
    print()

    print("Meaning:")
    print(f"  - Focal length: How 'zoomed in' the camera is")
    print(f"    fx ≈ fy = {(fx+fy)/2:.0f} → Aspect ratio is square")
    print(f"  - Principal point: Optical center of the sensor")
    print(f"    (cx, cy) = ({cx:.0f}, {cy:.0f})")
    print()

    # 2. Rotation Matrix (R)
    print("-" * 80)
    print("2. ROTATION MATRIX (R) - Camera Orientation")
    print("-" * 80)
    R = cam0['R']
    print(f"Shape: {R.shape}")
    print(f"Data type: {R.dtype}")
    print()
    print("Matrix:")
    print(R)
    print()

    print("Properties:")
    # Check orthogonality
    RRT = R @ R.T
    is_orthogonal = np.allclose(RRT, np.eye(3), atol=1e-6)
    print(f"  Orthogonal (R @ R^T = I): {is_orthogonal}")

    # Check determinant
    det = np.linalg.det(R)
    print(f"  Determinant: {det:.6f} (should be +1 for proper rotation)")
    print()

    # Extract axis directions
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    print("Camera axes in world coordinates:")
    print(f"  X-axis (right): [{x_axis[0]:+.3f}, {x_axis[1]:+.3f}, {x_axis[2]:+.3f}]")
    print(f"  Y-axis (down):  [{y_axis[0]:+.3f}, {y_axis[1]:+.3f}, {y_axis[2]:+.3f}]")
    print(f"  Z-axis (forward): [{z_axis[0]:+.3f}, {z_axis[1]:+.3f}, {z_axis[2]:+.3f}]")
    print()

    print("Meaning:")
    print(f"  - This matrix transforms world coordinates → camera coordinates")
    print(f"  - Camera's viewing direction: Z-axis")
    print(f"  - 'Up' in camera image: -Y-axis")
    print()

    # 3. Translation Vector (T)
    print("-" * 80)
    print("3. TRANSLATION VECTOR (T) - Camera Position")
    print("-" * 80)
    T = cam0['T']
    print(f"Shape: {T.shape}")
    print(f"Data type: {T.dtype}")
    print()
    print("Vector:")
    print(T)
    print()

    if T.ndim == 2:
        T_flat = T.flatten()
    else:
        T_flat = T

    print(f"Position in world coordinates:")
    print(f"  X: {T_flat[0]:+.4f} meters")
    print(f"  Y: {T_flat[1]:+.4f} meters")
    print(f"  Z: {T_flat[2]:+.4f} meters")
    print()

    # Calculate camera center
    # Camera center in world: C = -R^T @ T
    C = -R.T @ T_flat
    print(f"Camera center (C = -R^T @ T):")
    print(f"  X: {C[0]:+.4f} meters")
    print(f"  Y: {C[1]:+.4f} meters")
    print(f"  Z: {C[2]:+.4f} meters")
    print()

    # 4. Distortion maps (mapx, mapy)
    print("-" * 80)
    print("4. UNDISTORTION MAPS (mapx, mapy)")
    print("-" * 80)
    mapx = cam0['mapx']
    mapy = cam0['mapy']
    print(f"mapx shape: {mapx.shape}")
    print(f"mapy shape: {mapy.shape}")
    print(f"Data type: {mapx.dtype}")
    print()

    print("Meaning:")
    print(f"  - Pre-computed lookup tables for lens distortion correction")
    print(f"  - Each pixel (x, y) maps to undistorted (mapx[y,x], mapy[y,x])")
    print(f"  - Image size: {mapx.shape[1]} × {mapx.shape[0]} pixels")
    print()

    # Summary for all cameras
    print("=" * 80)
    print("ALL CAMERAS SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Cam':<4} {'fx':<8} {'fy':<8} {'cx':<8} {'cy':<8} {'Tx':<10} {'Ty':<10} {'Tz':<10}")
    print("-" * 80)

    for i, cam in enumerate(cameras):
        K = cam['K']
        T = cam['T']
        if T.ndim == 2:
            T = T.flatten()

        print(f"{i:<4} {K[0,0]:<8.1f} {K[1,1]:<8.1f} {K[0,2]:<8.1f} {K[1,2]:<8.1f} "
              f"{T[0]:<+10.4f} {T[1]:<+10.4f} {T[2]:<+10.4f}")

    print()

    # Camera positions visualization
    print("=" * 80)
    print("CAMERA SETUP GEOMETRY")
    print("=" * 80)
    print()

    camera_centers = []
    camera_directions = []

    for i, cam in enumerate(cameras):
        R = cam['R']
        T = cam['T']
        if T.ndim == 2:
            T = T.flatten()

        # Camera center in world
        C = -R.T @ T
        camera_centers.append(C)

        # Camera viewing direction (Z-axis)
        z_axis = R[:, 2]
        camera_directions.append(z_axis)

    camera_centers = np.array(camera_centers)
    camera_directions = np.array(camera_directions)

    # Calculate center of all cameras
    setup_center = camera_centers.mean(axis=0)

    print("Camera positions relative to setup center:")
    print(f"Setup center: [{setup_center[0]:+.4f}, {setup_center[1]:+.4f}, {setup_center[2]:+.4f}]")
    print()

    for i, C in enumerate(camera_centers):
        relative = C - setup_center
        distance = np.linalg.norm(relative)
        print(f"Camera {i}: distance = {distance:.4f}m, "
              f"position = [{C[0]:+.4f}, {C[1]:+.4f}, {C[2]:+.4f}]")

    print()

    # Check if cameras point inward
    print("Camera viewing directions (should point toward center):")
    for i, (C, direction) in enumerate(zip(camera_centers, camera_directions)):
        to_center = setup_center - C
        to_center_norm = to_center / np.linalg.norm(to_center)

        # Dot product: 1 = same direction, -1 = opposite
        alignment = np.dot(direction, to_center_norm)

        print(f"Camera {i}: alignment with center = {alignment:+.3f} "
              f"({'pointing inward' if alignment > 0.5 else 'not aligned'})")

    print()

    return cameras, camera_centers, camera_directions


def create_visualization(cameras, camera_centers, camera_directions, output_path=None):
    """
    Create 3D visualization of camera setup
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera positions
    ax.scatter(camera_centers[:, 0],
               camera_centers[:, 1],
               camera_centers[:, 2],
               c='red', s=100, marker='o', label='Cameras')

    # Plot camera viewing directions
    for i, (C, direction) in enumerate(zip(camera_centers, camera_directions)):
        ax.quiver(C[0], C[1], C[2],
                  direction[0], direction[1], direction[2],
                  length=0.2, color='blue', arrow_length_ratio=0.3)

        # Label cameras
        ax.text(C[0], C[1], C[2], f'  Cam{i}', fontsize=10)

    # Plot setup center
    setup_center = camera_centers.mean(axis=0)
    ax.scatter([setup_center[0]], [setup_center[1]], [setup_center[2]],
               c='green', s=200, marker='*', label='Setup Center')

    # Plot Y-axes (up direction) for each camera
    for i, cam in enumerate(cameras):
        R = cam['R']
        y_axis = R[:, 1]
        C = camera_centers[i]
        ax.quiver(C[0], C[1], C[2],
                  y_axis[0], y_axis[1], y_axis[2],
                  length=0.15, color='orange', arrow_length_ratio=0.3, alpha=0.6)

    # Set labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Multi-Camera Setup - Top View\n(Blue: viewing direction, Orange: camera Y-axis)')
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([
        camera_centers[:, 0].max() - camera_centers[:, 0].min(),
        camera_centers[:, 1].max() - camera_centers[:, 1].min(),
        camera_centers[:, 2].max() - camera_centers[:, 2].min()
    ]).max() / 2.0

    mid_x = (camera_centers[:, 0].max() + camera_centers[:, 0].min()) * 0.5
    mid_y = (camera_centers[:, 1].max() + camera_centers[:, 1].min()) * 0.5
    mid_z = (camera_centers[:, 2].max() + camera_centers[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def compare_pkl_vs_h5(pkl_path, h5_path):
    """
    Compare pickle and HDF5 formats
    """
    import h5py

    print("=" * 80)
    print("PKL vs HDF5 COMPARISON")
    print("=" * 80)
    print()

    # Load both
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    with h5py.File(h5_path, 'r') as f:
        h5_intrinsic = f['camera_parameters']['intrinsic'][:]
        h5_rotation = f['camera_parameters']['rotation'][:]
        h5_translation = f['camera_parameters']['translation'][:]

    # Verify conversion
    print("Verification:")
    for i in range(len(pkl_data)):
        K_pkl = pkl_data[i]['K']
        R_pkl = pkl_data[i]['R']
        T_pkl = pkl_data[i]['T'].flatten() if pkl_data[i]['T'].ndim == 2 else pkl_data[i]['T']

        K_h5 = h5_intrinsic[i]
        R_h5 = h5_rotation[i]
        T_h5 = h5_translation[i]

        k_match = np.allclose(K_pkl, K_h5)
        r_match = np.allclose(R_pkl, R_h5)
        t_match = np.allclose(T_pkl, T_h5)

        status = "✓" if (k_match and r_match and t_match) else "✗"
        print(f"Camera {i}: {status} (K:{k_match}, R:{r_match}, T:{t_match})")

    print()

    # File size comparison
    import os
    pkl_size = os.path.getsize(pkl_path) / (1024**2)
    h5_size = os.path.getsize(h5_path) / (1024**2)

    print(f"File sizes:")
    print(f"  PKL: {pkl_size:.2f} MB")
    print(f"  HDF5: {h5_size:.2f} MB")
    print(f"  Reduction: {(1 - h5_size/pkl_size)*100:.1f}%")
    print()

    print("Why HDF5 is better:")
    print("  ✓ Much smaller (no distortion maps needed)")
    print("  ✓ Faster to load (binary format)")
    print("  ✓ Language-independent (C, Python, MATLAB, etc.)")
    print("  ✓ Supports partial loading (don't need to load entire file)")
    print("  ✓ No security issues (pickle can execute arbitrary code)")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_camera_params.py <new_cam.pkl> [camera_params.h5]")
        print()
        print("Example:")
        print("  python analyze_camera_params.py data/markerless_mouse_1_nerf/new_cam.pkl")
        sys.exit(1)

    pkl_path = sys.argv[1]
    h5_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Analyze pickle file
    cameras, centers, directions = analyze_camera_params(pkl_path)

    # Create visualization
    output_vis = pkl_path.replace('.pkl', '_visualization.png')
    create_visualization(cameras, centers, directions, output_vis)

    # Compare with HDF5 if provided
    if h5_path:
        compare_pkl_vs_h5(pkl_path, h5_path)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
