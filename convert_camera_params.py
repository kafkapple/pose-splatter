"""
Convert camera parameters from pickle to HDF5 format for Pose Splatter.
"""

import pickle
import h5py
import numpy as np
import sys

def convert_camera_params(pkl_path, h5_path):
    """
    Convert camera parameters from pickle to HDF5.

    Expected pickle format: List of dicts with keys ['K', 'R', 'T', 'mapx', 'mapy']
    Output HDF5 format: /camera_parameters/{rotation, translation, intrinsic}
    """
    # Load pickle
    with open(pkl_path, 'rb') as f:
        cams = pickle.load(f)

    n_cameras = len(cams)
    print(f"Found {n_cameras} cameras")

    # Extract parameters
    intrinsics = []
    rotations = []
    translations = []

    for i, cam in enumerate(cams):
        K = cam['K']  # [3, 3] intrinsic matrix
        R = cam['R']  # [3, 3] rotation matrix
        T = cam['T']  # [3, 1] or [3,] translation vector

        intrinsics.append(K)
        rotations.append(R)

        # Ensure T is 1D
        if T.ndim == 2:
            T = T.flatten()
        translations.append(T)

        print(f"Camera {i}: K={K.shape}, R={R.shape}, T={T.shape}")

    # Stack arrays
    intrinsics = np.array(intrinsics)  # [C, 3, 3]
    rotations = np.array(rotations)    # [C, 3, 3]
    translations = np.array(translations)  # [C, 3]

    print(f"\nFinal shapes:")
    print(f"  Intrinsics: {intrinsics.shape}")
    print(f"  Rotations: {rotations.shape}")
    print(f"  Translations: {translations.shape}")

    # Save to HDF5
    with h5py.File(h5_path, 'w') as f:
        cam_group = f.create_group('camera_parameters')
        cam_group.create_dataset('intrinsic', data=intrinsics)
        cam_group.create_dataset('rotation', data=rotations)
        cam_group.create_dataset('translation', data=translations)

    print(f"\nSaved to: {h5_path}")

    # Verify
    print("\nVerification:")
    with h5py.File(h5_path, 'r') as f:
        print(f"  Keys: {list(f['camera_parameters'].keys())}")
        for key in f['camera_parameters'].keys():
            print(f"  {key}: {f['camera_parameters'][key].shape}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_camera_params.py <input.pkl> <output.h5>")
        sys.exit(1)

    convert_camera_params(sys.argv[1], sys.argv[2])
