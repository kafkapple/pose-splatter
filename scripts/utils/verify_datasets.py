#!/usr/bin/env python3
"""
Verify dataset frame counts and match them against config expectations

This script checks all existing datasets and their corresponding configs
to identify any mismatches in expected vs actual frame counts.
"""

import os
import json
import zarr
import h5py
from pathlib import Path

def get_frame_count_from_zarr(zarr_path):
    """Get frame count from zarr file"""
    try:
        z = zarr.open(zarr_path, 'r')
        return len(z['images'])
    except Exception as e:
        return None

def get_frame_count_from_h5(h5_path):
    """Get frame count from h5 file"""
    try:
        with h5py.File(h5_path, 'r') as f:
            return f['images'].shape[0]
    except Exception as e:
        return None

def calculate_expected_frames(frame_jump, baseline_frames=3600):
    """Calculate expected frames based on frame_jump"""
    return int(baseline_frames * (5.0 / frame_jump))

def verify_datasets():
    """Verify all datasets in the project"""
    print("=" * 80)
    print("DATASET VERIFICATION REPORT")
    print("=" * 80)
    print()

    # Base directories
    config_dir = Path("configs")
    data_base = Path("data")

    # Find all config files
    configs = list(config_dir.glob("markerless_mouse*.json"))

    if not configs:
        print("No config files found!")
        return

    results = []

    for config_path in sorted(configs):
        print(f"\nChecking: {config_path.name}")
        print("-" * 80)

        try:
            with open(config_path) as f:
                config = json.load(f)

            # Extract key parameters
            frame_jump = config.get('frame_jump', 'N/A')
            data_dir = config.get('data_directory', 'N/A')
            image_dir = config.get('image_directory', 'N/A')

            print(f"  Config frame_jump: {frame_jump}")
            print(f"  Data directory: {data_dir}")

            # Calculate expected frames
            if frame_jump != 'N/A':
                expected = calculate_expected_frames(frame_jump)
                print(f"  Expected frames: ~{expected}")
            else:
                expected = None
                print(f"  Expected frames: N/A (no frame_jump in config)")

            # Check actual dataset
            zarr_path = None
            h5_path = None
            actual_frames = None

            if image_dir != 'N/A':
                zarr_path = Path(image_dir) / "images.zarr"
                h5_path = Path(image_dir) / "images.h5"

            # Try zarr first
            if zarr_path and zarr_path.exists():
                actual_frames = get_frame_count_from_zarr(zarr_path)
                print(f"  Dataset found: {zarr_path}")
                print(f"  Actual frames: {actual_frames}")
            # Try h5 as fallback
            elif h5_path and h5_path.exists():
                actual_frames = get_frame_count_from_h5(h5_path)
                print(f"  Dataset found: {h5_path}")
                print(f"  Actual frames: {actual_frames}")
            else:
                print(f"  Dataset: NOT FOUND")
                print(f"    Searched: {zarr_path}, {h5_path}")

            # Verify match
            if expected and actual_frames:
                diff = abs(actual_frames - expected)
                match_status = "✓ MATCH" if diff <= 500 else "✗ MISMATCH"

                if diff > 500:
                    print(f"\n  ⚠️  WARNING: {match_status}")
                    print(f"     Difference: {diff} frames")
                    print(f"     This indicates data was likely generated with frame_jump={int(3600 * 5 / actual_frames)}")
                else:
                    print(f"\n  {match_status}")

                results.append({
                    'config': config_path.name,
                    'frame_jump': frame_jump,
                    'expected': expected,
                    'actual': actual_frames,
                    'diff': diff,
                    'match': diff <= 500
                })
            elif actual_frames is None:
                print(f"\n  ⚠️  MISSING DATA")
                results.append({
                    'config': config_path.name,
                    'frame_jump': frame_jump,
                    'expected': expected,
                    'actual': None,
                    'diff': None,
                    'match': False
                })

        except Exception as e:
            print(f"  Error processing config: {e}")
            continue

    # Summary
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if not results:
        print("No datasets to verify.")
        return

    print(f"{'Config':<40} {'FJ':<5} {'Expected':<10} {'Actual':<10} {'Status':<10}")
    print("-" * 80)

    for r in results:
        config_name = r['config'][:38]
        fj = str(r['frame_jump'])
        exp = f"~{r['expected']}" if r['expected'] else "N/A"
        act = str(r['actual']) if r['actual'] is not None else "MISSING"
        status = "✓" if r['match'] else "✗"

        print(f"{config_name:<40} {fj:<5} {exp:<10} {act:<10} {status:<10}")

    # Recommendations
    print("\n")
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    mismatches = [r for r in results if not r['match'] and r['actual'] is not None]
    missing = [r for r in results if r['actual'] is None]

    if mismatches:
        print("Datasets with mismatches:")
        for r in mismatches:
            print(f"  • {r['config']}")
            print(f"    Action: Regenerate with write_images.py using this config")
            print(f"    Command: python3 write_images.py configs/{r['config']}")
            print()

    if missing:
        print("Missing datasets:")
        for r in missing:
            print(f"  • {r['config']}")
            print(f"    Action: Generate data with write_images.py")
            print(f"    Command: python3 write_images.py configs/{r['config']}")
            print()

    if not mismatches and not missing:
        print("✓ All datasets validated successfully!")

if __name__ == '__main__':
    verify_datasets()
