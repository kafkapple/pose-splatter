"""
Compare multiple configuration files and highlight differences.
"""
__date__ = "November 2025"

import json
import argparse
from pathlib import Path
from tabulate import tabulate


def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def compare_configs(config_files):
    """Compare multiple config files and show differences."""
    configs = {}
    for config_file in config_files:
        name = Path(config_file).stem
        configs[name] = load_config(config_file)

    # Get all unique keys
    all_keys = set()
    for config in configs.values():
        all_keys.update(config.keys())
    all_keys = sorted(all_keys)

    # Build comparison table
    rows = []
    for key in all_keys:
        row = [key]
        values = []
        for name in sorted(configs.keys()):
            value = configs[name].get(key, "N/A")
            values.append(str(value))

        # Check if all values are the same
        if len(set(values)) > 1:
            row.extend(values)
            rows.append(row)

    headers = ["Parameter"] + sorted(configs.keys())

    return rows, headers


def main():
    parser = argparse.ArgumentParser(description="Compare config files")
    parser.add_argument("configs", nargs='+', help="Config JSON files to compare")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for comparison table")
    parser.add_argument("--format", type=str, default="grid",
                       choices=["grid", "markdown", "latex"],
                       help="Table format")

    args = parser.parse_args()

    print("Comparing configurations:")
    for config in args.configs:
        print(f"  - {config}")
    print()

    rows, headers = compare_configs(args.configs)

    if not rows:
        print("All configurations are identical!")
        return

    table = tabulate(rows, headers=headers, tablefmt=args.format)
    print("\nDifferences found:\n")
    print(table)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(table)
        print(f"\nComparison saved to: {args.output}")


if __name__ == '__main__':
    main()
