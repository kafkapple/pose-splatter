"""
Create organized export directory with timestamp and config name.
Organizes all visualization outputs into a downloadable package.
"""
import argparse
import os
import shutil
import json
from datetime import datetime
from pathlib import Path


def create_organized_export(config_path, base_output_dir, export_name=None, include_checkpoint=False):
    """
    Create organized export directory.

    Args:
        config_path: Path to config file
        base_output_dir: Base output directory (e.g., output/markerless_mouse_nerf)
        export_name: Custom export name (default: config_name_timestamp)
        include_checkpoint: Whether to copy checkpoint.pt (default: symlink)
    """

    # Get config name
    config_name = Path(config_path).stem

    # Generate export name with timestamp
    if export_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{config_name}_{timestamp}"

    # Create export directory
    export_dir = Path("exports") / export_name
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"========================================")
    print(f"Creating Organized Export")
    print(f"========================================")
    print(f"Config: {config_name}")
    print(f"Export Directory: {export_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    base_path = Path(base_output_dir)

    # Create manifest
    manifest = {
        "config_name": config_name,
        "config_path": str(config_path),
        "export_name": export_name,
        "timestamp": datetime.now().isoformat(),
        "contents": {}
    }

    # 1. Copy renders (videos and images)
    if (base_path / "renders").exists():
        print("Copying renders...")
        render_dest = export_dir / "renders"
        shutil.copytree(base_path / "renders", render_dest, dirs_exist_ok=True)
        manifest["contents"]["renders"] = {
            "path": "renders/",
            "description": "Rendered images and videos (multiview, temporal, rotation)"
        }

    # 2. Copy/link animations
    if (base_path / "animation").exists():
        print("Copying animation sequences...")
        anim_dest = export_dir / "animation"
        shutil.copytree(base_path / "animation", anim_dest, dirs_exist_ok=True)
        manifest["contents"]["animation"] = {
            "path": "animation/",
            "description": "Multi-frame animation sequences (PLY + NPZ)"
        }

    # 3. Copy point clouds
    if (base_path / "pointclouds").exists():
        print("Copying point clouds...")
        pc_dest = export_dir / "pointclouds"
        shutil.copytree(base_path / "pointclouds", pc_dest, dirs_exist_ok=True)
        manifest["contents"]["pointclouds"] = {
            "path": "pointclouds/",
            "description": "Single frame point clouds (PLY format)"
        }

    # 4. Copy Gaussian exports
    if (base_path / "gaussians").exists():
        print("Copying Gaussian exports...")
        gauss_dest = export_dir / "gaussians"
        shutil.copytree(base_path / "gaussians", gauss_dest, dirs_exist_ok=True)
        manifest["contents"]["gaussians"] = {
            "path": "gaussians/",
            "description": "Full Gaussian parameters (NPZ, JSON)"
        }

    # 5. Handle checkpoint.pt
    checkpoint_path = base_path / "checkpoint.pt"
    if checkpoint_path.exists():
        if include_checkpoint:
            print("Copying checkpoint.pt...")
            shutil.copy2(checkpoint_path, export_dir / "checkpoint.pt")
            manifest["contents"]["checkpoint"] = {
                "path": "checkpoint.pt",
                "description": "Model checkpoint (copied)",
                "size_mb": checkpoint_path.stat().st_size / (1024*1024)
            }
        else:
            print("Creating symlink to checkpoint.pt...")
            try:
                checkpoint_link = export_dir / "checkpoint.pt"
                if checkpoint_link.exists():
                    checkpoint_link.unlink()
                checkpoint_link.symlink_to(checkpoint_path.absolute())
                manifest["contents"]["checkpoint"] = {
                    "path": "checkpoint.pt",
                    "description": "Model checkpoint (symlink)",
                    "original": str(checkpoint_path.absolute())
                }
            except:
                print("  ⚠ Symlink failed, copying instead...")
                shutil.copy2(checkpoint_path, export_dir / "checkpoint.pt")

    # 6. Copy metrics if exists
    metrics_path = base_path / "metrics_test.csv"
    if metrics_path.exists():
        print("Copying metrics...")
        shutil.copy2(metrics_path, export_dir / "metrics_test.csv")
        manifest["contents"]["metrics"] = {
            "path": "metrics_test.csv",
            "description": "Evaluation metrics"
        }

    # 7. Copy loss plot if exists
    loss_path = base_path / "loss.pdf"
    if loss_path.exists():
        print("Copying loss plot...")
        shutil.copy2(loss_path, export_dir / "loss.pdf")
        manifest["contents"]["loss_plot"] = {
            "path": "loss.pdf",
            "description": "Training loss curve"
        }

    # 8. Copy config
    print("Copying config...")
    shutil.copy2(config_path, export_dir / "config.json")
    manifest["contents"]["config"] = {
        "path": "config.json",
        "description": "Training configuration"
    }

    # 9. Create README
    readme_content = f"""# {export_name}

Export created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Config: {config_name}

## Contents

"""
    for key, info in manifest["contents"].items():
        readme_content += f"### {key.title()}\n"
        readme_content += f"- **Path**: `{info['path']}`\n"
        readme_content += f"- **Description**: {info['description']}\n"
        if 'size_mb' in info:
            readme_content += f"- **Size**: {info['size_mb']:.1f} MB\n"
        readme_content += "\n"

    readme_content += """## Usage

### View Point Clouds
```bash
# Using MeshLab, CloudCompare, or Blender
meshlab pointclouds/frame0000.ply
```

### View Videos
```bash
# Temporal sequence
mpv renders/temporal/temporal_view0_frames0-60.mp4

# 360 rotation
mpv renders/rotation360/rotation360.mp4
```

### Load in Blender
```python
# Open blender_import_pointcloud.py and edit paths
# Then run in Blender (Alt+P)
PLY_FILE = "pointclouds/frame0000.ply"
```

### Analyze Gaussian Parameters
```python
import numpy as np

# Load NPZ file
data = np.load('gaussians/gaussian_frame0000.npz')
means = data['means']
colors = data['colors']
opacities = data['opacities']
scales = data['scales']
quaternions = data['quaternions']
```

## File Structure

See manifest.json for complete file listing.
"""

    with open(export_dir / "README.md", 'w') as f:
        f.write(readme_content)

    # 10. Save manifest
    with open(export_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # 11. Calculate total size
    total_size = sum(
        f.stat().st_size
        for f in export_dir.rglob('*')
        if f.is_file() and not f.is_symlink()
    )

    print("")
    print("========================================")
    print("✓ Export Complete!")
    print("========================================")
    print(f"Export Directory: {export_dir}")
    print(f"Total Size: {total_size / (1024*1024):.1f} MB")
    print(f"Files: {sum(1 for _ in export_dir.rglob('*') if _.is_file())}")
    print("")
    print("To create downloadable archive:")
    print(f"  tar -czf {export_name}.tar.gz -C exports {export_name}")
    print(f"  zip -r {export_name}.zip exports/{export_name}")
    print("")

    return export_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create organized export directory")
    parser.add_argument("--config", type=str, default="configs/markerless_mouse_nerf.json",
                       help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="output/markerless_mouse_nerf",
                       help="Base output directory")
    parser.add_argument("--name", type=str, default=None,
                       help="Custom export name (default: config_timestamp)")
    parser.add_argument("--copy_checkpoint", action="store_true",
                       help="Copy checkpoint.pt instead of creating symlink")

    args = parser.parse_args()

    create_organized_export(
        args.config,
        args.output_dir,
        export_name=args.name,
        include_checkpoint=args.copy_checkpoint
    )
