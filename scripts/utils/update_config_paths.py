#!/usr/bin/env python3
"""
Update all config files to use relative paths instead of absolute paths.

This script:
1. Finds all JSON config files
2. Converts absolute paths to relative paths
3. Backs up original configs
4. Updates configs in place

Usage:
    python scripts/utils/update_config_paths.py [--dry-run] [--backup]
"""

import json
import os
import sys
import argparse
from pathlib import Path
import shutil

def get_project_root():
    """Get project root directory"""
    # Assume script is in scripts/utils/
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent

def make_path_relative(abs_path, project_root):
    """
    Convert absolute path to relative path from project root

    Examples:
        /home/joon/dev/pose-splatter/data/... → data/...
        /home/joon/pose-splatter/data/... → data/...
    """
    abs_path = str(abs_path).rstrip('/')

    # Check if already relative
    if not os.path.isabs(abs_path):
        return abs_path

    # Try to make relative to project root
    try:
        abs_path_obj = Path(abs_path)
        project_root_obj = Path(project_root)

        # Check if path is under project root
        try:
            rel_path = abs_path_obj.relative_to(project_root_obj)
            return str(rel_path)
        except ValueError:
            # Path is not under project root
            pass

        # Check if path contains known patterns
        parts = abs_path.split('/')

        # Look for 'data/' or 'output/' in path
        if 'data' in parts:
            idx = parts.index('data')
            return '/'.join(parts[idx:])

        if 'output' in parts:
            idx = parts.index('output')
            return '/'.join(parts[idx:])

        # If can't convert, return original
        print(f"  Warning: Could not make relative: {abs_path}")
        return abs_path

    except Exception as e:
        print(f"  Warning: Error converting {abs_path}: {e}")
        return abs_path

def update_config(config_path, project_root, dry_run=False, backup=True):
    """
    Update a single config file
    """
    print(f"\nProcessing: {config_path}")

    # Read config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Track changes
    changes = []

    # Update data_directory
    if 'data_directory' in config:
        old_val = config['data_directory']
        new_val = make_path_relative(old_val, project_root)
        if old_val != new_val:
            changes.append(('data_directory', old_val, new_val))
            config['data_directory'] = new_val

    # Update project_directory
    if 'project_directory' in config:
        old_val = config['project_directory']
        new_val = make_path_relative(old_val, project_root)
        if old_val != new_val:
            changes.append(('project_directory', old_val, new_val))
            config['project_directory'] = new_val

    # Print changes
    if changes:
        print(f"  Changes:")
        for key, old, new in changes:
            print(f"    {key}:")
            print(f"      Old: {old}")
            print(f"      New: {new}")
    else:
        print(f"  No changes needed (already relative)")
        return False

    # Save if not dry run
    if not dry_run:
        # Backup original
        if backup:
            backup_path = str(config_path) + '.bak'
            shutil.copy2(config_path, backup_path)
            print(f"  Backup: {backup_path}")

        # Write updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"  ✓ Updated")
    else:
        print(f"  (dry-run, not saved)")

    return True

def main():
    parser = argparse.ArgumentParser(description='Update config paths to relative')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Backup original configs (default: True)')
    parser.add_argument('--no-backup', action='store_false', dest='backup',
                       help='Do not backup original configs')
    args = parser.parse_args()

    print("=" * 80)
    print("Config Path Updater")
    print("=" * 80)
    print()

    # Get project root
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    print()

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print()

    # Find all config files
    config_dir = project_root / 'configs'
    config_files = list(config_dir.rglob('*.json'))

    print(f"Found {len(config_files)} config files")
    print()

    # Update each config
    updated_count = 0
    for config_path in sorted(config_files):
        if update_config(config_path, project_root,
                        dry_run=args.dry_run,
                        backup=args.backup):
            updated_count += 1

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total configs: {len(config_files)}")
    print(f"Updated: {updated_count}")
    print(f"Unchanged: {len(config_files) - updated_count}")

    if args.dry_run:
        print()
        print("This was a dry run. To apply changes, run without --dry-run:")
        print(f"  python {sys.argv[0]} --backup")
    else:
        print()
        print("✓ All configs updated successfully!")
        if args.backup:
            print("  Backups saved with .bak extension")

if __name__ == '__main__':
    main()
