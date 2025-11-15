#!/usr/bin/env python3
"""
Test 2D/3D Gaussian renderer with actual checkpoint
"""
import sys
import torch
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PoseSplatter


def load_config(config_path):
    """Load config from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def test_checkpoint_loading(checkpoint_path, config_path, mode="3d"):
    """Test loading checkpoint and running inference"""
    print(f"\n{'='*60}")
    print(f"Testing checkpoint with {mode.upper()} renderer")
    print(f"{'='*60}\n")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"✓ Checkpoint loaded")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'loss' in checkpoint:
        loss_val = checkpoint['loss']
        if isinstance(loss_val, (int, float)):
            print(f"  - Loss: {loss_val:.4f}")
        else:
            # It's a list, print the last value
            if isinstance(loss_val, list) and len(loss_val) > 0:
                last_loss = loss_val[-1]
                if isinstance(last_loss, list):
                    print(f"  - Loss (last epoch): {last_loss}")
                else:
                    print(f"  - Loss: {last_loss:.4f}")
    else:
        print("  - Loss: N/A")

    # Load config from JSON
    print(f"Loading config: {config_path}")
    json_config = load_config(config_path)
    print(f"✓ Config loaded")

    # Use dummy camera params (actual params don't matter for testing renderer)
    C = 6  # Number of cameras
    W = json_config.get('image_width', 2048) // json_config.get('image_downsample', 4)
    H = json_config.get('image_height', 1536) // json_config.get('image_downsample', 4)

    # Create dummy intrinsics and extrinsics
    intrinsics = []
    extrinsics = []
    for i in range(C):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = W / 2  # focal length
        K[0, 2] = W / 2  # cx
        K[1, 2] = H / 2  # cy
        intrinsics.append(K)

        E = np.eye(4)
        E[0, 3] = i * 0.1  # Spread cameras around
        extrinsics.append(E)

    intrinsics = np.array(intrinsics)
    extrinsics = np.array(extrinsics)

    print(f"  - Image size: {W}x{H}")
    print(f"  - Num cameras: {C}")

    # Create model with specified mode
    model = PoseSplatter(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        W=W,
        H=H,
        device="cuda",
        volume_idx=json_config.get('volume_idx', None),
        grid_size=json_config.get('grid_size', 64),
        gaussian_mode=mode,
        gaussian_config={},
    ).to("cuda")

    print(f"✓ Created model with {mode} renderer")
    print(f"  - Renderer type: {type(model.renderer).__name__}")
    print(f"  - Num params: {model.renderer.get_num_params()}")

    # Load state dict (only compatible parameters)
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']

    # Filter out incompatible keys
    compatible_state = {}
    skipped_keys = []
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    model.load_state_dict(compatible_state, strict=False)
    print(f"✓ Loaded state dict")
    print(f"  - Compatible params: {len(compatible_state)}")
    print(f"  - Skipped params: {len(skipped_keys)}")
    if skipped_keys:
        print(f"  - Skipped: {skipped_keys[:3]}..." if len(skipped_keys) > 3 else f"  - Skipped: {skipped_keys}")

    # Set to eval mode
    model.eval()

    # Create dummy inputs (use actual data shape)
    C = len(intrinsics)
    mask = (torch.rand(1, C, H, W, device='cpu') > 0.5).float()  # Binary mask as float
    img = torch.rand(1, C, 3, H, W, device='cpu')                # RGB images
    p_3d = torch.zeros(3, device='cpu')                          # Center position
    angle = 0.0                                                   # No rotation

    print(f"\n✓ Created dummy inputs")
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Image shape: {img.shape}")

    # Run inference
    print(f"\nRunning inference...")
    with torch.no_grad():
        try:
            rgb, alpha = model(mask, img, p_3d, angle, view_num=0)
            print(f"✅ Inference successful!")
            print(f"  - RGB shape: {rgb.shape}")
            print(f"  - Alpha shape: {alpha.shape}")
            print(f"  - RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
            print(f"  - Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
            return True
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main test function"""
    checkpoint_path = "output/markerless_mouse_nerf_extended/checkpoint.pt"
    config_path = "configs/markerless_mouse_nerf_extended.json"

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    if not Path(config_path).exists():
        print(f"Error: Config not found at {config_path}")
        return

    # Test both 3D and 2D modes
    success_3d = test_checkpoint_loading(checkpoint_path, config_path, mode="3d")
    success_2d = test_checkpoint_loading(checkpoint_path, config_path, mode="2d")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"3D mode: {'✅ PASSED' if success_3d else '❌ FAILED'}")
    print(f"2D mode: {'✅ PASSED' if success_2d else '❌ FAILED'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
