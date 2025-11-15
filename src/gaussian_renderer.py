"""
Gaussian Renderer Module

Provides unified interface for 2D and 3D Gaussian splatting rendering.
Supports switching between rendering modes via config.

Classes:
    GaussianRenderer: Abstract base class
    GaussianRenderer2D: 2D Gaussian splatting implementation
    GaussianRenderer3D: 3D Gaussian splatting implementation (gsplat)

Functions:
    create_renderer: Factory function to create renderer instances
"""
__date__ = "November 2025"

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn


class GaussianRenderer(ABC, nn.Module):
    """
    Abstract base class for Gaussian renderers.

    Provides unified interface for 2D and 3D Gaussian splatting.
    All concrete implementations must implement get_num_params() and render().

    Attributes:
        width (int): Output image width
        height (int): Output image height
        device (str): Device for computation ("cuda" or "cpu")
        background_color (torch.Tensor): Background RGB color [3]
    """

    def __init__(self, width: int, height: int, device: str = "cuda"):
        """
        Initialize GaussianRenderer.

        Args:
            width: Output image width
            height: Output image height
            device: Device for computation ("cuda" or "cpu")
        """
        super().__init__()
        self.width = width
        self.height = height
        self.device = device

        # Register background color as buffer (not trainable)
        # Default to black background for cleaner visualization
        self.register_buffer(
            'background_color',
            torch.zeros(3, device=device)
        )

    @abstractmethod
    def get_num_params(self) -> int:
        """
        Return number of parameters per Gaussian.

        Returns:
            Number of parameters per Gaussian point
        """
        pass

    @abstractmethod
    def render(
        self,
        gaussian_params: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render Gaussians to image.

        Args:
            gaussian_params: Gaussian parameters [N, P] where P = get_num_params()
                            Each row contains parameters for one Gaussian
            viewmat: Camera view matrix [4, 4]
                    World-to-camera transformation
            K: Camera intrinsic matrix [3, 3]
               [[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]]

        Returns:
            rgb: Rendered RGB image [H, W, 3], values in [0, 1]
            alpha: Alpha channel [H, W], values in [0, 1]

        Note:
            Some implementations (e.g., 2D) may not use viewmat or K.
            They are included for interface consistency.
        """
        pass

    def set_background_color(self, color: torch.Tensor):
        """
        Set background color.

        Args:
            color: RGB color [3], values in [0, 1]
        """
        if color.shape != (3,):
            raise ValueError(f"Expected color shape (3,), got {color.shape}")
        self.background_color.copy_(color.to(self.device))


class GaussianRenderer3D(GaussianRenderer):
    """
    3D Gaussian Splatting renderer using gsplat library.

    Parameters per Gaussian: 14
    Layout:
        - [:, 0:3]: means (x, y, z) in world coordinates
        - [:, 3:6]: log_scales (log of scale in each axis)
        - [:, 6:10]: quats (quaternion for rotation, will be normalized)
        - [:, 10:13]: colors (r, g, b) in [0, 1]
        - [:, 13]: opacities (logit, will be passed through sigmoid)

    Reference:
        "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
        Kerbl et al., 2023
    """

    def __init__(self, width: int, height: int, device: str = "cuda"):
        """
        Initialize 3D Gaussian Renderer.

        Args:
            width: Output image width
            height: Output image height
            device: Device for computation ("cuda" or "cpu")
        """
        super().__init__(width, height, device)

        # Import gsplat (only when this renderer is used)
        try:
            from gsplat.rendering import rasterization
            self.rasterization = rasterization
        except ImportError as e:
            raise ImportError(
                "gsplat library is required for 3D rendering. "
                "Install with: pip install gsplat"
            ) from e

    def get_num_params(self) -> int:
        """
        Return number of parameters per Gaussian.

        Returns:
            14 (means:3 + scales:3 + quats:4 + colors:3 + opacities:1)
        """
        return 14

    def render(
        self,
        gaussian_params: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render 3D Gaussians using gsplat.

        Args:
            gaussian_params: [N, 14] Gaussian parameters
            viewmat: [4, 4] view matrix
            K: [3, 3] intrinsic matrix

        Returns:
            rgb: [H, W, 3] RGB image
            alpha: [H, W] alpha channel
        """
        if gaussian_params.shape[1] != 14:
            raise ValueError(
                f"Expected 14 parameters per Gaussian, got {gaussian_params.shape[1]}"
            )

        N = gaussian_params.shape[0]

        # Parse parameters
        means = gaussian_params[:, 0:3]                    # [N, 3]
        log_scales = gaussian_params[:, 3:6]               # [N, 3]
        quats = gaussian_params[:, 6:10]                   # [N, 4]
        colors = gaussian_params[:, 10:13]                 # [N, 3]
        logit_opacities = gaussian_params[:, 13:14]        # [N, 1]

        # Apply activations
        scales = torch.exp(log_scales)                     # Exponential for scales
        quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize quats
        colors = torch.clamp(colors, 0.0, 1.0)             # Clamp colors to [0, 1]
        opacities = torch.sigmoid(logit_opacities).squeeze(-1)  # Sigmoid for opacities [N]

        # Rasterize using gsplat (returns rgb, alpha, meta)
        rgb, alpha, meta = self.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],                        # [1, 4, 4]
            Ks=K[None],                                    # [1, 3, 3]
            width=self.width,
            height=self.height,
            packed=False,
            backgrounds=self.background_color[None],       # [1, 3]
        )

        # Remove batch dimension and return
        return rgb[0], alpha[0, ..., 0]


class GaussianRenderer2D(GaussianRenderer):
    """
    2D Gaussian Splatting renderer.

    Parameters per Gaussian: 9
    Layout:
        - [:, 0:2]: means_2d (u, v) in pixel coordinates
        - [:, 2:4]: log_scales_2d (log of scale in each axis, in pixels)
        - [:, 4]: rotation (angle in radians)
        - [:, 5:8]: colors (r, g, b) in [0, 1]
        - [:, 8]: opacities (logit, will be passed through sigmoid)

    Note:
        This renderer operates directly in 2D image space without 3D-to-2D projection.
        It does not use viewmat or K, but accepts them for interface consistency.

    Reference:
        "2D Gaussian Splatting for Geometrically Accurate Radiance Fields"
        Huang et al., 2024
    """

    def __init__(
        self,
        width: int,
        height: int,
        device: str = "cuda",
        kernel_size: int = 5,
        sigma_cutoff: float = 3.0,
    ):
        """
        Initialize 2D Gaussian Renderer.

        Args:
            width: Output image width
            height: Output image height
            device: Device for computation ("cuda" or "cpu")
            kernel_size: Size of Gaussian kernel (not currently used)
            sigma_cutoff: Number of standard deviations for Gaussian cutoff
        """
        super().__init__(width, height, device)
        self.kernel_size = kernel_size
        self.sigma_cutoff = sigma_cutoff

    def get_num_params(self) -> int:
        """
        Return number of parameters per Gaussian.

        Returns:
            9 (means_2d:2 + scales_2d:2 + rotation:1 + colors:3 + opacities:1)
        """
        return 9

    def render(
        self,
        gaussian_params: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render 2D Gaussians.

        Args:
            gaussian_params: [N, 9] Gaussian parameters
            viewmat: [4, 4] view matrix (not used, for interface consistency)
            K: [3, 3] intrinsic matrix (not used, for interface consistency)

        Returns:
            rgb: [H, W, 3] RGB image
            alpha: [H, W] alpha channel

        Note:
            This is a reference implementation using sequential splatting.
            For production, consider vectorized or CUDA implementation.
        """
        if gaussian_params.shape[1] != 9:
            raise ValueError(
                f"Expected 9 parameters per Gaussian, got {gaussian_params.shape[1]}"
            )

        N = gaussian_params.shape[0]

        # Parse parameters
        means_2d = gaussian_params[:, 0:2]                 # [N, 2]
        log_scales_2d = gaussian_params[:, 2:4]            # [N, 2]
        rotation = gaussian_params[:, 4]                   # [N]
        colors = gaussian_params[:, 5:8]                   # [N, 3]
        logit_opacities = gaussian_params[:, 8]            # [N]

        # Apply activations
        scales_2d = torch.exp(log_scales_2d)               # Exponential for scales
        colors = torch.clamp(colors, 0.0, 1.0)             # Clamp colors
        opacities = torch.sigmoid(logit_opacities)         # Sigmoid for opacities

        # Initialize canvas (requires_grad=True to enable backprop)
        canvas = torch.zeros(
            (self.height, self.width, 3),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True
        )
        alpha_canvas = torch.zeros(
            (self.height, self.width),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True
        )

        # Sort Gaussians by depth (for proper alpha blending)
        # For 2D, we can use y-coordinate or keep original order
        # Here we keep original order for simplicity

        # Splat each Gaussian (accumulate contributions)
        for i in range(N):
            canvas, alpha_canvas = self._splat_gaussian_2d(
                canvas,
                alpha_canvas,
                means_2d[i],
                scales_2d[i],
                rotation[i],
                colors[i],
                opacities[i],
            )

        # Composite with background color
        # canvas contains the accumulated color contribution from Gaussians
        # background fills in where alpha is low (transmittance is high)
        transmittance = 1.0 - alpha_canvas.unsqueeze(-1)  # [H, W, 1]
        final_rgb = canvas + transmittance * self.background_color.view(1, 1, 3)

        return final_rgb, alpha_canvas

    def _splat_gaussian_2d(
        self,
        canvas: torch.Tensor,
        alpha_canvas: torch.Tensor,
        mean_2d: torch.Tensor,
        scale_2d: torch.Tensor,
        rotation: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splat a single 2D Gaussian onto canvas.

        Uses rotated elliptical Gaussian kernel with alpha blending.

        Args:
            canvas: [H, W, 3] RGB canvas
            alpha_canvas: [H, W] alpha canvas
            mean_2d: [2] center position (u, v)
            scale_2d: [2] scale (sx, sy)
            rotation: scalar rotation angle in radians
            color: [3] RGB color
            opacity: scalar opacity

        Returns:
            canvas: [H, W, 3] Updated RGB canvas
            alpha_canvas: [H, W] Updated alpha canvas
        """
        u, v = mean_2d
        sx, sy = scale_2d
        theta = rotation

        # Compute bounding box (3-sigma cutoff)
        radius = max(sx, sy) * self.sigma_cutoff
        u_min = int(torch.clamp(u - radius, 0, self.width - 1).item())
        u_max = int(torch.clamp(u + radius, 0, self.width - 1).item())
        v_min = int(torch.clamp(v - radius, 0, self.height - 1).item())
        v_max = int(torch.clamp(v + radius, 0, self.height - 1).item())

        # Check if Gaussian is out of bounds
        if u_max <= u_min or v_max <= v_min:
            return canvas, alpha_canvas

        # Create grid for the bounding box
        y_grid, x_grid = torch.meshgrid(
            torch.arange(v_min, v_max + 1, device=self.device, dtype=torch.float32),
            torch.arange(u_min, u_max + 1, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

        # Compute displacement from center
        dx = x_grid - u
        dy = y_grid - v

        # Apply rotation
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx_rot = cos_theta * dx + sin_theta * dy
        dy_rot = -sin_theta * dx + cos_theta * dy

        # Compute Gaussian weights
        # G(x, y) = exp(-(dx_rot^2 / (2*sx^2) + dy_rot^2 / (2*sy^2)))
        gauss = torch.exp(
            -(dx_rot**2 / (2 * sx**2 + 1e-8) + dy_rot**2 / (2 * sy**2 + 1e-8))
        )
        gauss = gauss * opacity  # Apply opacity

        # Get current transmittance
        alpha_slice = alpha_canvas[v_min:v_max+1, u_min:u_max+1]
        transmittance = 1.0 - alpha_slice

        # Compute contribution with alpha blending
        contribution = gauss * transmittance

        # Clone canvas to avoid in-place modification
        new_canvas = canvas.clone()
        new_alpha_canvas = alpha_canvas.clone()

        # Update canvas (front-to-back blending) - NON in-place
        for c in range(3):
            new_canvas[v_min:v_max+1, u_min:u_max+1, c] = (
                canvas[v_min:v_max+1, u_min:u_max+1, c] + contribution * color[c]
            )

        # Update alpha canvas - NON in-place
        new_alpha_canvas[v_min:v_max+1, u_min:u_max+1] = (
            alpha_canvas[v_min:v_max+1, u_min:u_max+1] + contribution
        )

        return new_canvas, new_alpha_canvas


def create_renderer(
    mode: str,
    width: int,
    height: int,
    device: str = "cuda",
    **kwargs
) -> GaussianRenderer:
    """
    Factory function to create Gaussian renderer.

    Args:
        mode: Rendering mode, either "2d" or "3d"
        width: Output image width
        height: Output image height
        device: Device for computation ("cuda" or "cpu")
        **kwargs: Additional renderer-specific arguments
                 For 2D: kernel_size, sigma_cutoff
                 For 3D: (none currently)

    Returns:
        GaussianRenderer instance (either 2D or 3D)

    Raises:
        ValueError: If mode is not "2d" or "3d"

    Examples:
        >>> # Create 3D renderer
        >>> renderer = create_renderer("3d", 256, 256, device="cuda")
        >>>
        >>> # Create 2D renderer with custom sigma cutoff
        >>> renderer = create_renderer("2d", 256, 256, device="cuda", sigma_cutoff=4.0)
    """
    mode = mode.lower()

    if mode == "2d":
        return GaussianRenderer2D(width, height, device, **kwargs)
    elif mode == "3d":
        return GaussianRenderer3D(width, height, device)
    else:
        raise ValueError(
            f"Unknown renderer mode: '{mode}'. Expected '2d' or '3d'."
        )


# Utility functions for parameter conversion (future work)
def convert_3d_to_2d_params(
    params_3d: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """
    Convert 3D Gaussian parameters to 2D by projection.

    This is useful for initializing 2D renderer from 3D checkpoint
    or for hybrid rendering approaches.

    Args:
        params_3d: [N, 14] 3D Gaussian parameters
        viewmat: [4, 4] camera view matrix
        K: [3, 3] camera intrinsic matrix

    Returns:
        params_2d: [N, 9] 2D Gaussian parameters

    Note:
        This is a placeholder for future implementation.
        Requires careful handling of covariance projection.
    """
    raise NotImplementedError("3D to 2D parameter conversion not yet implemented")


def convert_2d_to_3d_params(
    params_2d: torch.Tensor,
    depth: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """
    Convert 2D Gaussian parameters to 3D by unprojection.

    This requires depth estimation or assumption.

    Args:
        params_2d: [N, 9] 2D Gaussian parameters
        depth: [N] depth value for each Gaussian
        viewmat: [4, 4] camera view matrix
        K: [3, 3] camera intrinsic matrix

    Returns:
        params_3d: [N, 14] 3D Gaussian parameters

    Note:
        This is a placeholder for future implementation.
    """
    raise NotImplementedError("2D to 3D parameter conversion not yet implemented")
