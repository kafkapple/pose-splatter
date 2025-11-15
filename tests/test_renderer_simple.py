"""
Simple test script for Gaussian Renderer (no pytest required).
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.gaussian_renderer import (
    GaussianRenderer2D,
    GaussianRenderer3D,
    create_renderer
)


def test_2d_basic():
    """Test 2D renderer basic functionality."""
    print("\n=== Test 2D Renderer Basic ===")

    renderer = GaussianRenderer2D(256, 256, device="cpu")
    print(f"✓ Created 2D renderer: {renderer.width}x{renderer.height}")
    print(f"✓ Num params: {renderer.get_num_params()}")

    # Create a red Gaussian at center
    params = torch.tensor([[
        128.0, 128.0,  # mean (center)
        1.0, 1.0,      # log_scale
        0.0,           # rotation
        1.0, 0.0, 0.0, # color (red)
        2.0            # logit_opacity
    ]])

    rgb, alpha = renderer.render(params, None, None)
    print(f"✓ Rendered: rgb {rgb.shape}, alpha {alpha.shape}")

    # Check center is red
    center_color = rgb[128, 128]
    print(f"✓ Center color: R={center_color[0]:.3f}, G={center_color[1]:.3f}, B={center_color[2]:.3f}")

    assert center_color[0] > 0.5, "Center should be red"
    assert center_color[1] < 0.1, "Center should not be green"
    print("✓ Center is red as expected")

    # Check center has high alpha
    center_alpha = alpha[128, 128]
    print(f"✓ Center alpha: {center_alpha:.3f}")
    assert center_alpha > 0.5, "Center should have high alpha"

    print("✅ 2D basic test PASSED\n")


def test_2d_multiple():
    """Test 2D renderer with multiple Gaussians."""
    print("\n=== Test 2D Multiple Gaussians ===")

    renderer = GaussianRenderer2D(256, 256, device="cpu")

    # Red at (64, 128), Blue at (192, 128)
    params = torch.tensor([
        [64.0, 128.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0],  # Red
        [192.0, 128.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0],  # Blue
    ])

    rgb, alpha = renderer.render(params, None, None)

    # Check red Gaussian
    red_pixel = rgb[128, 64]
    print(f"✓ Red Gaussian (64, 128): R={red_pixel[0]:.3f}, B={red_pixel[2]:.3f}")
    assert red_pixel[0] > 0.5, "Left should be red"
    assert red_pixel[2] < 0.1, "Left should not be blue"

    # Check blue Gaussian
    blue_pixel = rgb[128, 192]
    print(f"✓ Blue Gaussian (192, 128): R={blue_pixel[0]:.3f}, B={blue_pixel[2]:.3f}")
    assert blue_pixel[2] > 0.5, "Right should be blue"
    assert blue_pixel[0] < 0.1, "Right should not be red"

    print("✅ 2D multiple Gaussians test PASSED\n")


def test_factory():
    """Test factory function."""
    print("\n=== Test Factory Function ===")

    # Create 2D
    renderer_2d = create_renderer("2d", 256, 256, device="cpu")
    print(f"✓ Created via factory: {type(renderer_2d).__name__}")
    assert isinstance(renderer_2d, GaussianRenderer2D)
    assert renderer_2d.get_num_params() == 9
    print(f"✓ 2D renderer has {renderer_2d.get_num_params()} params")

    # Test case insensitivity
    renderer_2d_upper = create_renderer("2D", 256, 256, device="cpu")
    assert type(renderer_2d_upper) == type(renderer_2d)
    print("✓ Case insensitive mode works")

    # Test kwargs
    renderer_custom = create_renderer(
        "2d", 256, 256, device="cpu",
        sigma_cutoff=4.0,
        kernel_size=7
    )
    assert renderer_custom.sigma_cutoff == 4.0
    print("✓ Kwargs forwarding works")

    print("✅ Factory function test PASSED\n")


def test_3d_basic():
    """Test 3D renderer (if gsplat available)."""
    print("\n=== Test 3D Renderer Basic ===")

    try:
        renderer = GaussianRenderer3D(256, 256, device="cpu")
        print(f"✓ Created 3D renderer: {renderer.width}x{renderer.height}")
        print(f"✓ Num params: {renderer.get_num_params()}")
        assert renderer.get_num_params() == 14

        print("✅ 3D basic test PASSED (gsplat available)\n")
    except ImportError as e:
        print(f"⚠️  3D test SKIPPED: {e}\n")


def test_background_color():
    """Test background color setting."""
    print("\n=== Test Background Color ===")

    renderer = GaussianRenderer2D(256, 256, device="cpu")

    # Set blue background
    renderer.set_background_color(torch.tensor([0.0, 0.0, 1.0]))
    print("✓ Set background to blue")

    # Render empty
    params = torch.zeros((0, 9))
    rgb, alpha = renderer.render(params, None, None)

    corner_color = rgb[0, 0]
    print(f"✓ Corner color: R={corner_color[0]:.3f}, G={corner_color[1]:.3f}, B={corner_color[2]:.3f}")

    assert torch.allclose(corner_color, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    print("✓ Background is blue")

    print("✅ Background color test PASSED\n")


def test_invalid_params():
    """Test error handling."""
    print("\n=== Test Error Handling ===")

    renderer = GaussianRenderer2D(256, 256, device="cpu")

    # Wrong number of parameters
    params = torch.randn(10, 14)  # Should be 9

    try:
        rgb, alpha = renderer.render(params, None, None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    print("✅ Error handling test PASSED\n")


def main():
    """Run all tests."""
    print("="*60)
    print("GAUSSIAN RENDERER TESTS")
    print("="*60)

    tests = [
        test_2d_basic,
        test_2d_multiple,
        test_factory,
        test_3d_basic,
        test_background_color,
        test_invalid_params,
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
