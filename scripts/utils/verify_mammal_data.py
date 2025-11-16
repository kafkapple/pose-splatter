#!/usr/bin/env python3
"""
Verify MAMMAL_mouse dataset structure and file integrity

This script checks that all required files from the MAMMAL_mouse dataset
have been correctly copied to the data directory.
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and return its size"""
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {description}: {size_mb:.1f} MB")
        return True, size_mb
    else:
        print(f"  ✗ {description}: NOT FOUND")
        return False, 0

def verify_mammal_dataset(data_dir="data/markerless_mouse_1_nerf"):
    """Verify MAMMAL_mouse dataset structure"""
    print("=" * 80)
    print("MAMMAL_mouse Dataset Verification")
    print("=" * 80)
    print()

    data_path = Path(data_dir)

    # Check if directory exists
    if not data_path.exists():
        print(f"✗ ERROR: Data directory not found: {data_path}")
        print()
        print("Please copy data from MAMMAL_mouse repository:")
        print("  cp -r ~/dev/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \\")
        print(f"        {data_dir}/")
        return False

    print(f"Data directory: {data_path.resolve()}")
    print()

    all_checks_passed = True
    total_size = 0

    # Check videos_undist folder
    print("1. RGB Videos (videos_undist/):")
    videos_dir = data_path / "videos_undist"
    if videos_dir.exists():
        expected_videos = [f"{i}.mp4" for i in range(6)]
        for video in expected_videos:
            video_path = videos_dir / video
            passed, size = check_file_exists(video_path, f"Camera {video[0]}")
            all_checks_passed &= passed
            total_size += size
    else:
        print(f"  ✗ Directory not found: {videos_dir}")
        all_checks_passed = False
    print()

    # Check simpleclick_undist folder
    print("2. Segmentation Masks (simpleclick_undist/):")
    masks_dir = data_path / "simpleclick_undist"
    if masks_dir.exists():
        expected_masks = [f"{i}.mp4" for i in range(6)]
        for mask in expected_masks:
            mask_path = masks_dir / mask
            passed, size = check_file_exists(mask_path, f"Mask {mask[0]}")
            all_checks_passed &= passed
            total_size += size
    else:
        print(f"  ✗ Directory not found: {masks_dir}")
        all_checks_passed = False
    print()

    # Check keypoints2d_undist folder
    print("3. 2D Keypoints (keypoints2d_undist/):")
    keypoints_dir = data_path / "keypoints2d_undist"
    if keypoints_dir.exists():
        expected_keypoints = [f"result_view_{i}.pkl" for i in range(6)]
        for kp in expected_keypoints:
            kp_path = keypoints_dir / kp
            passed, size = check_file_exists(kp_path, f"View {kp.split('_')[2].split('.')[0]}")
            all_checks_passed &= passed
            total_size += size
    else:
        print(f"  ✗ Directory not found: {keypoints_dir}")
        all_checks_passed = False
    print()

    # Check camera calibration
    print("4. Camera Calibration:")
    cam_path = data_path / "new_cam.pkl"
    passed, size = check_file_exists(cam_path, "Camera parameters")
    all_checks_passed &= passed
    total_size += size
    print()

    # Check 3D keypoint labels
    print("5. 3D Keypoint Labels:")
    labels_path = data_path / "add_labels_3d_8keypoints.pkl"
    passed, size = check_file_exists(labels_path, "3D keypoint labels")
    all_checks_passed &= passed
    total_size += size
    print()

    # Check label IDs
    print("6. Label IDs:")
    label_ids_path = data_path / "label_ids.pkl"
    passed, size = check_file_exists(label_ids_path, "Label ID mapping")
    all_checks_passed &= passed
    total_size += size
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total dataset size: {total_size:.1f} MB (expected: ~299 MB)")
    print()

    if all_checks_passed:
        print("✓ All required files found!")
        print()
        print("You can now proceed with preprocessing:")
        print("  bash scripts/preprocessing/run_full_preprocessing.sh \\")
        print("       configs/baseline/markerless_mouse_nerf.json")
        return True
    else:
        print("✗ Some files are missing!")
        print()
        print("To fix this, copy the complete dataset from MAMMAL_mouse:")
        print()
        print("Option 1: Copy from local MAMMAL_mouse repository")
        print("  cp -r ~/dev/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \\")
        print(f"        {data_dir}/")
        print()
        print("Option 2: Clone MAMMAL_mouse and copy")
        print("  git clone https://github.com/kafkapple/MAMMAL_mouse.git /tmp/MAMMAL_mouse")
        print("  cp -r /tmp/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/* \\")
        print(f"        {data_dir}/")
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/markerless_mouse_1_nerf"

    success = verify_mammal_dataset(data_dir)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
