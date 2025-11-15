"""
Integration test for PoseSplatter model with 2D/3D renderer support.

Tests that the model can be instantiated and run inference with both
2D and 3D rendering modes.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.model import PoseSplatter


def create_dummy_inputs(H=256, W=256, C=6, device="cpu"):
    """Create dummy inputs for model testing."""
    # Dummy camera parameters
    intrinsics = np.array([np.eye(3) * 256 for _ in range(C)])
    intrinsics[:, 0, 2] = W // 2
    intrinsics[:, 1, 2] = H // 2

    extrinsics = np.array([np.eye(4) for _ in range(C)])

    # Dummy inputs
    mask = torch.rand(1, C, H, W)  # [1, C, H, W]
    img = torch.rand(1, C, 3, H, W)  # [1, C, 3, H, W]
    p_3d = torch.tensor([0.0, 0.0, 0.0])
    angle = 0.0

    return intrinsics, extrinsics, mask, img, p_3d, angle


def test_model_3d_mode():
    """Test model with 3D rendering mode."""
    print("\n=== Test PoseSplatter with 3D Renderer ===")

    H, W = 64, 64  # Small size for quick test
    intrinsics, extrinsics, mask, img, p_3d, angle = create_dummy_inputs(H, W, device="cpu")

    # Create model
    model = PoseSplatter(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        W=W,
        H=H,
        device="cuda",
        volume_idx=[[0, 16], [0, 16], [0, 16]],  # Small volume
        grid_size=16,
        gaussian_mode="3d",
        gaussian_config={},
    ).to("cuda")

    print(f"✓ Created model: gaussian_mode={model.gaussian_mode}")
    print(f"✓ Renderer type: {type(model.renderer).__name__}")
    print(f"✓ Num gaussian params: {model.renderer.get_num_params()}")

    # Run inference (set to eval mode to avoid BatchNorm issues with small volumes)
    model.eval()
    with torch.no_grad():
        rgb, alpha = model(mask, img, p_3d, angle, view_num=0)

    print(f"✓ Forward pass successful")
    print(f"✓ RGB shape: {rgb.shape}, expected: [1, {H}, {W}, 3]")
    print(f"✓ Alpha shape: {alpha.shape}, expected: [1, {H}, {W}, 1]")

    assert rgb.shape == (1, H, W, 3), f"Wrong RGB shape: {rgb.shape}"
    assert alpha.shape == (1, H, W, 1), f"Wrong alpha shape: {alpha.shape}"

    print("✅ 3D mode test PASSED\n")


def test_model_2d_mode():
    """Test model with 2D rendering mode."""
    print("\n=== Test PoseSplatter with 2D Renderer ===")

    H, W = 64, 64
    intrinsics, extrinsics, mask, img, p_3d, angle = create_dummy_inputs(H, W, device="cpu")

    # Create model
    model = PoseSplatter(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        W=W,
        H=H,
        device="cuda",
        volume_idx=[[0, 16], [0, 16], [0, 16]],
        grid_size=16,
        gaussian_mode="2d",
        gaussian_config={"sigma_cutoff": 3.0},
    ).to("cuda")

    print(f"✓ Created model: gaussian_mode={model.gaussian_mode}")
    print(f"✓ Renderer type: {type(model.renderer).__name__}")
    print(f"✓ Num gaussian params: {model.renderer.get_num_params()}")

    # Run inference (set to eval mode to avoid BatchNorm issues with small volumes)
    model.eval()
    with torch.no_grad():
        rgb, alpha = model(mask, img, p_3d, angle, view_num=0)

    print(f"✓ Forward pass successful")
    print(f"✓ RGB shape: {rgb.shape}")
    print(f"✓ Alpha shape: {alpha.shape}")

    assert rgb.shape == (1, H, W, 3), f"Wrong RGB shape: {rgb.shape}"
    assert alpha.shape == (1, H, W, 1), f"Wrong alpha shape: {alpha.shape}"

    print("✅ 2D mode test PASSED\n")


def test_parameter_count():
    """Test that parameter count matches renderer mode."""
    print("\n=== Test Parameter Count ===")

    H, W = 64, 64
    intrinsics, extrinsics, _, _, _, _ = create_dummy_inputs(H, W, device="cpu")

    # 3D model
    model_3d = PoseSplatter(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        W=W,
        H=H,
        device="cpu",
        volume_idx=[[0, 16], [0, 16], [0, 16]],
        grid_size=16,
        gaussian_mode="3d",
    )

    # Check MLP output size
    mlp_output_size_3d = model_3d.gaussian_param_net[-1].out_features
    print(f"✓ 3D MLP output size: {mlp_output_size_3d} (expected 14)")
    assert mlp_output_size_3d == 14

    # 2D model
    model_2d = PoseSplatter(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        W=W,
        H=H,
        device="cpu",
        volume_idx=[[0, 16], [0, 16], [0, 16]],
        grid_size=16,
        gaussian_mode="2d",
    )

    mlp_output_size_2d = model_2d.gaussian_param_net[-1].out_features
    print(f"✓ 2D MLP output size: {mlp_output_size_2d} (expected 9)")
    assert mlp_output_size_2d == 9

    print("✅ Parameter count test PASSED\n")


def test_background_color():
    """Test that background color is properly set."""
    print("\n=== Test Background Color ===")

    H, W = 64, 64
    intrinsics, extrinsics, _, _, _, _ = create_dummy_inputs(H, W, device="cpu")

    model = PoseSplatter(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        W=W,
        H=H,
        device="cpu",
        volume_idx=[[0, 16], [0, 16], [0, 16]],
        grid_size=16,
        gaussian_mode="2d",
    )

    # Check background color
    print(f"✓ Model background: {model.background_color}")
    print(f"✓ Renderer background: {model.renderer.background_color}")

    assert torch.allclose(model.background_color, model.renderer.background_color)

    print("✅ Background color test PASSED\n")


def main():
    """Run all integration tests."""
    print("="*60)
    print("POSE SPLATTER MODEL INTEGRATION TESTS")
    print("="*60)

    tests = [
        test_model_3d_mode,
        test_model_2d_mode,
        test_parameter_count,
        test_background_color,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("="*60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
