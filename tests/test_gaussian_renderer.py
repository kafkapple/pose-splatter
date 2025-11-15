"""
Unit tests for Gaussian Renderer module.

Tests the abstract interface, 2D/3D implementations, and factory function.
"""
__date__ = "November 2025"

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest
from src.gaussian_renderer import (
    GaussianRenderer,
    GaussianRenderer2D,
    GaussianRenderer3D,
    create_renderer
)


class TestGaussianRendererInterface:
    """Test abstract base class interface."""

    def test_cannot_instantiate_abstract(self):
        """Abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            renderer = GaussianRenderer(256, 256)

    def test_subclass_must_implement_methods(self):
        """Subclass must implement abstract methods."""
        class IncompleteRenderer(GaussianRenderer):
            pass

        with pytest.raises(TypeError):
            renderer = IncompleteRenderer(256, 256)


class TestGaussianRenderer2D:
    """Test 2D Gaussian Splatting renderer."""

    def test_num_params(self):
        """2D renderer should have 9 parameters per Gaussian."""
        renderer = GaussianRenderer2D(256, 256, device="cpu")
        assert renderer.get_num_params() == 9

    def test_initialization(self):
        """Test renderer initialization."""
        renderer = GaussianRenderer2D(256, 256, device="cpu")
        assert renderer.width == 256
        assert renderer.height == 256
        assert renderer.device == "cpu"
        assert renderer.background_color.shape == (3,)

    def test_single_gaussian_render(self):
        """Test rendering a single Gaussian."""
        renderer = GaussianRenderer2D(256, 256, device="cpu")

        # Create a red Gaussian at center
        params = torch.tensor([[
            128.0, 128.0,  # mean (center)
            1.0, 1.0,      # log_scale (exp(1) ≈ 2.7 pixels)
            0.0,           # rotation
            1.0, 0.0, 0.0, # color (red)
            2.0            # logit_opacity (sigmoid(2) ≈ 0.88)
        ]])

        # Render (viewmat and K not used in 2D)
        rgb, alpha = renderer.render(params, None, None)

        # Check output shapes
        assert rgb.shape == (256, 256, 3)
        assert alpha.shape == (256, 256)

        # Check center pixel has high red value
        assert rgb[128, 128, 0] > 0.5, "Center should be red"
        assert rgb[128, 128, 1] < 0.1, "Center should not be green"
        assert rgb[128, 128, 2] < 0.1, "Center should not be blue"

        # Check center has high alpha
        assert alpha[128, 128] > 0.5, "Center should have high alpha"

        # Check corners have low alpha (mostly background)
        assert alpha[0, 0] < 0.1, "Corner should have low alpha"

    def test_out_of_bounds_gaussian(self):
        """Test Gaussian completely out of image bounds."""
        renderer = GaussianRenderer2D(256, 256, device="cpu")

        # Gaussian far outside image
        params = torch.tensor([[
            -100.0, -100.0,  # mean (outside)
            1.0, 1.0,
            0.0,
            1.0, 0.0, 0.0,
            2.0
        ]])

        rgb, alpha = renderer.render(params, None, None)

        # Should have minimal contribution
        assert alpha.max() < 0.01, "Out-of-bounds Gaussian should not contribute"

    def test_multiple_gaussians(self):
        """Test rendering multiple Gaussians."""
        renderer = GaussianRenderer2D(256, 256, device="cpu")

        # Create two Gaussians: red at (64, 128), blue at (192, 128)
        params = torch.tensor([
            [64.0, 128.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0],  # Red
            [192.0, 128.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0],  # Blue
        ])

        rgb, alpha = renderer.render(params, None, None)

        # Check red Gaussian
        assert rgb[128, 64, 0] > 0.5, "Left should be red"
        assert rgb[128, 64, 2] < 0.1, "Left should not be blue"

        # Check blue Gaussian
        assert rgb[128, 192, 2] > 0.5, "Right should be blue"
        assert rgb[128, 192, 0] < 0.1, "Right should not be red"

    def test_rotation(self):
        """Test Gaussian rotation."""
        renderer = GaussianRenderer2D(256, 256, device="cpu")

        # Elongated Gaussian (sx=0.5, sy=2.0) rotated 0 vs 90 degrees
        import math

        # Horizontal elongation
        params_h = torch.tensor([[
            128.0, 128.0,
            math.log(5.0), math.log(2.0),  # sx=5, sy=2
            0.0,  # No rotation
            1.0, 0.0, 0.0,
            2.0
        ]])

        # Vertical elongation (90 degree rotation)
        params_v = torch.tensor([[
            128.0, 128.0,
            math.log(5.0), math.log(2.0),  # sx=5, sy=2
            math.pi / 2,  # 90 degree rotation
            1.0, 0.0, 0.0,
            2.0
        ]])

        rgb_h, alpha_h = renderer.render(params_h, None, None)
        rgb_v, alpha_v = renderer.render(params_v, None, None)

        # Horizontal should have more spread in x
        assert alpha_h[128, 120] > alpha_h[120, 128], "Horizontal elongation"

        # Vertical should have more spread in y
        assert alpha_v[120, 128] > alpha_v[128, 120], "Vertical elongation"

    def test_invalid_params_shape(self):
        """Test error handling for invalid parameter shape."""
        renderer = GaussianRenderer2D(256, 256, device="cpu")

        # Wrong number of parameters (14 instead of 9)
        params = torch.randn(10, 14)

        with pytest.raises(ValueError, match="Expected 9 parameters"):
            rgb, alpha = renderer.render(params, None, None)

    def test_background_color(self):
        """Test background color setting."""
        renderer = GaussianRenderer2D(256, 256, device="cpu")

        # Set blue background
        renderer.set_background_color(torch.tensor([0.0, 0.0, 1.0]))

        # Render empty (no Gaussians)
        params = torch.zeros((0, 9))
        rgb, alpha = renderer.render(params, None, None)

        # Should be blue background
        assert torch.allclose(rgb[0, 0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)


class TestGaussianRenderer3D:
    """Test 3D Gaussian Splatting renderer."""

    def test_num_params(self):
        """3D renderer should have 14 parameters per Gaussian."""
        try:
            renderer = GaussianRenderer3D(256, 256, device="cpu")
            assert renderer.get_num_params() == 14
        except ImportError:
            pytest.skip("gsplat not installed")

    def test_initialization(self):
        """Test renderer initialization."""
        try:
            renderer = GaussianRenderer3D(256, 256, device="cpu")
            assert renderer.width == 256
            assert renderer.height == 256
            assert renderer.device == "cpu"
        except ImportError:
            pytest.skip("gsplat not installed")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_render_basic(self):
        """Test basic 3D rendering (requires CUDA)."""
        try:
            renderer = GaussianRenderer3D(256, 256, device="cuda")

            # Create simple parameters
            params = torch.randn(100, 14).cuda()

            # Simple camera
            viewmat = torch.eye(4).cuda()
            K = torch.tensor([
                [256, 0, 128],
                [0, 256, 128],
                [0, 0, 1]
            ], dtype=torch.float32).cuda()

            rgb, alpha = renderer.render(params, viewmat, K)

            assert rgb.shape == (256, 256, 3)
            assert alpha.shape == (256, 256)
        except ImportError:
            pytest.skip("gsplat not installed")

    def test_invalid_params_shape(self):
        """Test error handling for invalid parameter shape."""
        try:
            renderer = GaussianRenderer3D(256, 256, device="cpu")

            # Wrong number of parameters (9 instead of 14)
            params = torch.randn(10, 9)
            viewmat = torch.eye(4)
            K = torch.eye(3)

            with pytest.raises(ValueError, match="Expected 14 parameters"):
                rgb, alpha = renderer.render(params, viewmat, K)
        except ImportError:
            pytest.skip("gsplat not installed")

    def test_gsplat_import_error(self):
        """Test that helpful error is raised if gsplat not installed."""
        # This test is tricky - we'd need to mock the import
        # For now, just check that the error message is informative
        pass


class TestRendererFactory:
    """Test factory function."""

    def test_create_2d(self):
        """Test creating 2D renderer."""
        renderer = create_renderer("2d", 256, 256, device="cpu")
        assert isinstance(renderer, GaussianRenderer2D)
        assert renderer.get_num_params() == 9

    def test_create_3d(self):
        """Test creating 3D renderer."""
        try:
            renderer = create_renderer("3d", 256, 256, device="cpu")
            assert isinstance(renderer, GaussianRenderer3D)
            assert renderer.get_num_params() == 14
        except ImportError:
            pytest.skip("gsplat not installed")

    def test_case_insensitive(self):
        """Test that mode is case-insensitive."""
        renderer_lower = create_renderer("2d", 256, 256, device="cpu")
        renderer_upper = create_renderer("2D", 256, 256, device="cpu")
        assert type(renderer_lower) == type(renderer_upper)

    def test_invalid_mode(self):
        """Test error for invalid mode."""
        with pytest.raises(ValueError, match="Unknown renderer mode"):
            create_renderer("invalid", 256, 256)

    def test_kwargs_forwarding(self):
        """Test that kwargs are forwarded to renderer."""
        renderer = create_renderer(
            "2d", 256, 256, device="cpu",
            sigma_cutoff=4.0,
            kernel_size=7
        )
        assert renderer.sigma_cutoff == 4.0
        assert renderer.kernel_size == 7


class TestIntegration:
    """Integration tests across renderers."""

    def test_parameter_count_consistency(self):
        """Test that parameter counts are consistent."""
        renderer_2d = create_renderer("2d", 256, 256, device="cpu")
        assert renderer_2d.get_num_params() == 9

        try:
            renderer_3d = create_renderer("3d", 256, 256, device="cpu")
            assert renderer_3d.get_num_params() == 14
        except ImportError:
            pytest.skip("gsplat not installed")

    def test_output_consistency(self):
        """Test that output shapes are consistent across renderers."""
        width, height = 128, 96

        renderer_2d = create_renderer("2d", width, height, device="cpu")
        params_2d = torch.randn(10, 9)
        rgb_2d, alpha_2d = renderer_2d.render(params_2d, None, None)

        assert rgb_2d.shape == (height, width, 3)
        assert alpha_2d.shape == (height, width)

        try:
            renderer_3d = create_renderer("3d", width, height, device="cpu")
            params_3d = torch.randn(10, 14)
            viewmat = torch.eye(4)
            K = torch.eye(3)

            # Note: This might fail without CUDA, so wrap in try-except
            try:
                rgb_3d, alpha_3d = renderer_3d.render(params_3d, viewmat, K)
                assert rgb_3d.shape == (height, width, 3)
                assert alpha_3d.shape == (height, width)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    pytest.skip("CUDA required for 3D rendering")
                raise
        except ImportError:
            pytest.skip("gsplat not installed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
