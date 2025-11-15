# 2D/3D Gaussian Splatting 전환 모듈 설계

**날짜**: 2025-11-12
**프로젝트**: pose-splatter
**목적**: 2D와 3D Gaussian Splatting 간 전환 가능한 모듈화 아키텍처 설계

---

## 1. 배경 및 동기

### 1.1 현재 상태

**3D Gaussian Splatting 사용**:
- `gsplat` 라이브러리의 `rasterization` 함수 사용
- 3D 공간에서 Gaussian을 정의하고 2D 이미지로 투영
- 14개 파라미터: means(3), scales(3), quats(4), colors(3), opacities(1)

**코드 위치**:
```python
# src/model.py:10-11
from gsplat.rendering import rasterization

# src/model.py:130-170
rgb_normalized, alpha = rasterization(
    means=gs_means,
    quats=gs_quats,
    scales=gs_scales,
    opacities=gs_opacities,
    colors=gs_colors,
    viewmats=viewmats[None],
    Ks=Ks[None],
    width=self.W,
    height=self.H,
    packed=False,
    backgrounds=self.background_color[None],
)
```

### 1.2 2D Gaussian Splatting의 필요성

**장점**:
1. **메모리 효율성**: 2D는 파라미터 수가 적음 (9개 vs 14개)
2. **계산 속도**: 3D → 2D 투영 단계 생략
3. **안정성**: Depth ambiguity 제거
4. **특정 응용**: Top-down view, fixed-plane tracking

**단점**:
1. **3D 일관성**: Multi-view consistency 보장 어려움
2. **유연성**: 새로운 viewpoint synthesis 제한적
3. **품질**: 3D 정보 손실

**Use Cases**:
- 2D: Pose tracking, 2D behavior analysis, overhead views
- 3D: Novel view synthesis, 3D reconstruction, volumetric capture

---

## 2. 설계 원칙

### 2.1 Core Principles

1. **모듈화 (Modularity)**: Renderer를 독립적 컴포넌트로 분리
2. **확장성 (Extensibility)**: 새로운 renderer 추가 용이
3. **일관성 (Consistency)**: 동일한 인터페이스로 2D/3D 접근
4. **성능 (Performance)**: 최소한의 오버헤드
5. **Config 기반 (Config-Driven)**: 코드 변경 없이 전환 가능

### 2.2 Non-Goals

- ❌ Real-time ray tracing
- ❌ Hybrid 2D/3D rendering (한 번에 하나만)
- ❌ Custom CUDA kernel 구현 (기존 라이브러리 활용)

---

## 3. 아키텍처 설계

### 3.1 Overall Architecture

```
┌─────────────────────────────────────────────────┐
│             PoseSplatter Model                  │
│                                                 │
│  ┌──────────────┐    ┌──────────────────────┐  │
│  │  UNet 3D     │───▶│ Gaussian Param Net   │  │
│  │  (features)  │    │ (Linear layers)      │  │
│  └──────────────┘    └──────────────────────┘  │
│                                │                │
│                                ▼                │
│                      ┌──────────────────┐       │
│                      │ GaussianRenderer │       │
│                      │   (Abstract)     │       │
│                      └──────────────────┘       │
│                                │                │
│                 ┌──────────────┴─────────────┐  │
│                 ▼                            ▼  │
│      ┌──────────────────┐      ┌──────────────────┐
│      │ 2D Renderer      │      │ 3D Renderer      │
│      │ (Custom impl)    │      │ (gsplat)         │
│      └──────────────────┘      └──────────────────┘
│                 │                            │  │
│                 └──────────────┬─────────────┘  │
│                                ▼                │
│                         ┌─────────────┐         │
│                         │  rgb, alpha │         │
│                         └─────────────┘         │
└─────────────────────────────────────────────────┘
```

### 3.2 Class Hierarchy

```python
# src/gaussian_renderer.py (NEW FILE)

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional

class GaussianRenderer(ABC, nn.Module):
    """
    Abstract base class for Gaussian renderers.

    Provides unified interface for 2D and 3D Gaussian splatting.
    """

    def __init__(self, width: int, height: int, device: str = "cuda"):
        super().__init__()
        self.width = width
        self.height = height
        self.device = device
        self.background_color = torch.ones(3).to(device)

    @abstractmethod
    def get_num_params(self) -> int:
        """Return number of parameters per Gaussian."""
        pass

    @abstractmethod
    def render(
        self,
        gaussian_params: torch.Tensor,  # [N, P] where P = num_params
        viewmat: torch.Tensor,          # [4, 4]
        K: torch.Tensor,                # [3, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render Gaussians to image.

        Args:
            gaussian_params: Gaussian parameters [N, P]
            viewmat: Camera view matrix [4, 4]
            K: Camera intrinsic matrix [3, 3]

        Returns:
            rgb: Rendered RGB image [H, W, 3]
            alpha: Alpha channel [H, W]
        """
        pass


class GaussianRenderer3D(GaussianRenderer):
    """
    3D Gaussian Splatting renderer using gsplat library.

    Parameters per Gaussian: 14
    - means: 3 (x, y, z)
    - scales: 3 (sx, sy, sz)
    - quats: 4 (rotation quaternion)
    - colors: 3 (r, g, b)
    - opacities: 1
    """

    def __init__(self, width: int, height: int, device: str = "cuda"):
        super().__init__(width, height, device)
        from gsplat.rendering import rasterization
        self.rasterization = rasterization

    def get_num_params(self) -> int:
        return 14  # 3 + 3 + 4 + 3 + 1

    def render(
        self,
        gaussian_params: torch.Tensor,  # [N, 14]
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render 3D Gaussians using gsplat.

        gaussian_params layout:
        - [:, 0:3]: means (x, y, z)
        - [:, 3:6]: scales (log scale)
        - [:, 6:10]: quats (normalized)
        - [:, 10:13]: colors (RGB)
        - [:, 13]: opacities (sigmoid)
        """
        N = gaussian_params.shape[0]

        # Parse parameters
        means = gaussian_params[:, 0:3]
        scales = torch.exp(gaussian_params[:, 3:6])
        quats = gaussian_params[:, 6:10]
        quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
        colors = gaussian_params[:, 10:13]
        opacities = torch.sigmoid(gaussian_params[:, 13:14])

        # Rasterize
        rgb, alpha = self.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=self.width,
            height=self.height,
            packed=False,
            backgrounds=self.background_color[None],
        )

        return rgb[0], alpha[0, ..., 0]


class GaussianRenderer2D(GaussianRenderer):
    """
    2D Gaussian Splatting renderer.

    Parameters per Gaussian: 9
    - means_2d: 2 (u, v in pixel space)
    - scales_2d: 2 (sx, sy in pixel space)
    - rotation: 1 (angle in radians)
    - colors: 3 (r, g, b)
    - opacities: 1

    Note: This operates directly in image space without 3D-to-2D projection.
    """

    def __init__(
        self,
        width: int,
        height: int,
        device: str = "cuda",
        kernel_size: int = 5,
    ):
        super().__init__(width, height, device)
        self.kernel_size = kernel_size

    def get_num_params(self) -> int:
        return 9  # 2 + 2 + 1 + 3 + 1

    def render(
        self,
        gaussian_params: torch.Tensor,  # [N, 9]
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render 2D Gaussians.

        gaussian_params layout:
        - [:, 0:2]: means_2d (u, v)
        - [:, 2:4]: scales_2d (log scale)
        - [:, 4]: rotation (radians)
        - [:, 5:8]: colors (RGB)
        - [:, 8]: opacities (sigmoid)

        Note: viewmat and K are accepted for interface consistency
              but not used in 2D rendering.
        """
        N = gaussian_params.shape[0]

        # Parse parameters
        means_2d = gaussian_params[:, 0:2]  # [N, 2]
        scales_2d = torch.exp(gaussian_params[:, 2:4])  # [N, 2]
        rotation = gaussian_params[:, 4]  # [N]
        colors = gaussian_params[:, 5:8]  # [N, 3]
        opacities = torch.sigmoid(gaussian_params[:, 8])  # [N]

        # Initialize canvas
        canvas = torch.zeros(
            (self.height, self.width, 3),
            device=self.device,
            dtype=torch.float32
        )
        alpha_canvas = torch.zeros(
            (self.height, self.width),
            device=self.device,
            dtype=torch.float32
        )

        # Splat each Gaussian
        for i in range(N):
            self._splat_gaussian_2d(
                canvas,
                alpha_canvas,
                means_2d[i],
                scales_2d[i],
                rotation[i],
                colors[i],
                opacities[i],
            )

        return canvas, alpha_canvas

    def _splat_gaussian_2d(
        self,
        canvas: torch.Tensor,
        alpha_canvas: torch.Tensor,
        mean_2d: torch.Tensor,
        scale_2d: torch.Tensor,
        rotation: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ):
        """
        Splat a single 2D Gaussian onto canvas.

        Uses rotated elliptical Gaussian kernel.
        """
        u, v = mean_2d
        sx, sy = scale_2d
        theta = rotation

        # Compute bounding box
        radius = max(sx, sy) * 3.0  # 3-sigma cutoff
        u_min = int(torch.clamp(u - radius, 0, self.width - 1))
        u_max = int(torch.clamp(u + radius, 0, self.width - 1))
        v_min = int(torch.clamp(v - radius, 0, self.height - 1))
        v_max = int(torch.clamp(v + radius, 0, self.height - 1))

        if u_max <= u_min or v_max <= v_min:
            return

        # Create grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(v_min, v_max + 1, device=self.device),
            torch.arange(u_min, u_max + 1, device=self.device),
            indexing='ij'
        )

        # Compute Gaussian weights
        dx = x_grid - u
        dy = y_grid - v

        # Apply rotation
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx_rot = cos_theta * dx + sin_theta * dy
        dy_rot = -sin_theta * dx + cos_theta * dy

        # Compute Gaussian
        gauss = torch.exp(
            -(dx_rot**2 / (2 * sx**2) + dy_rot**2 / (2 * sy**2))
        )
        gauss = gauss * opacity

        # Alpha blending
        alpha_slice = alpha_canvas[v_min:v_max+1, u_min:u_max+1]
        transmittance = 1.0 - alpha_slice
        contribution = gauss * transmittance

        # Update canvas
        for c in range(3):
            canvas[v_min:v_max+1, u_min:u_max+1, c] += (
                contribution * color[c]
            )
        alpha_canvas[v_min:v_max+1, u_min:u_max+1] += contribution


def create_renderer(
    mode: str,
    width: int,
    height: int,
    device: str = "cuda",
    **kwargs
) -> GaussianRenderer:
    """
    Factory function to create renderer.

    Args:
        mode: "2d" or "3d"
        width: Image width
        height: Image height
        device: Device ("cuda" or "cpu")
        **kwargs: Additional renderer-specific arguments

    Returns:
        GaussianRenderer instance
    """
    if mode == "2d":
        return GaussianRenderer2D(width, height, device, **kwargs)
    elif mode == "3d":
        return GaussianRenderer3D(width, height, device)
    else:
        raise ValueError(f"Unknown renderer mode: {mode}")
```

---

## 4. Model 통합

### 4.1 수정된 PoseSplatter

```python
# src/model.py (MODIFICATIONS)

from .gaussian_renderer import create_renderer

class PoseSplatter(nn.Module):
    def __init__(
            self,
            # ... existing params ...
            gaussian_mode: str = "3d",  # NEW PARAMETER
            gaussian_config: Optional[dict] = None,  # NEW PARAMETER
        ):
        super(PoseSplatter, self).__init__()

        # ... existing init code ...

        # Create renderer
        self.gaussian_mode = gaussian_mode
        self.renderer = create_renderer(
            mode=gaussian_mode,
            width=W,
            height=H,
            device=device,
            **(gaussian_config or {})
        )

        # Update Gaussian parameter network output size
        num_params = self.renderer.get_num_params()
        self.gaussian_param_net = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_params),  # CHANGED from 14
        )

        # ... rest of init ...

    def forward(self, mask, img, p_3d, angle, view_num=None):
        """
        Forward pass.

        Returns:
            rgb: [H, W, 3]
            alpha: [H, W]
        """
        # ... existing feature extraction code ...

        # Get current camera parameters
        if view_num is None:
            view_num = self.observed_views[0]

        viewmat = self.viewmats[view_num]
        K = self.Ks[view_num]

        # Extract Gaussian parameters
        gaussian_params = self.gaussian_param_net(features)  # [N, P]

        # Render using selected renderer
        rgb, alpha = self.renderer.render(
            gaussian_params,
            viewmat,
            K
        )

        return rgb, alpha
```

### 4.2 Config 수정

```json
// configs/markerless_mouse_nerf_2d.json (NEW FILE)
{
    "data_directory": "/home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf/",
    "project_directory": "/home/joon/dev/pose-splatter/output/markerless_mouse_nerf_2d/",

    // ... existing params ...

    "gaussian_mode": "2d",
    "gaussian_config": {
        "kernel_size": 5
    },

    // ... rest of config ...
}
```

```json
// configs/markerless_mouse_nerf_3d.json (EXISTING, ADD THIS)
{
    // ... existing params ...

    "gaussian_mode": "3d",
    "gaussian_config": {},

    // ... rest of config ...
}
```

---

## 5. 구현 단계

### Phase 1: 인터페이스 정의 (비-GPU, 1-2시간)

**작업**:
- [x] `src/gaussian_renderer.py` 생성
- [x] `GaussianRenderer` abstract base class 정의
- [x] Factory function `create_renderer()` 구현
- [x] Docstring 및 타입 힌트 작성

**산출물**:
- `src/gaussian_renderer.py` (abstract base only)
- 단위 테스트 스켈레톤

**검증**:
```python
from src.gaussian_renderer import GaussianRenderer

# Check interface
assert hasattr(GaussianRenderer, 'render')
assert hasattr(GaussianRenderer, 'get_num_params')
```

### Phase 2: 3D Renderer 리팩토링 (GPU 필요, 2-3시간)

**작업**:
- [ ] `GaussianRenderer3D` 구현
- [ ] 기존 `src/model.py`의 gsplat 코드 이전
- [ ] `PoseSplatter` 모델에 renderer 통합
- [ ] 기존 3D 학습과 동일한 결과 확인

**산출물**:
- 완성된 `GaussianRenderer3D` 클래스
- 수정된 `src/model.py`

**검증**:
```bash
# 기존 checkpoint로 inference 실행
python render_image.py \
  --config configs/markerless_mouse_nerf_extended.json \
  --checkpoint output/markerless_mouse_nerf_extended/checkpoint.pt \
  --output test_3d_refactor.png

# 결과 비교 (pixel-wise difference < 1e-5)
```

### Phase 3: 2D Renderer 구현 (GPU 필요, 4-6시간)

**작업**:
- [ ] `GaussianRenderer2D` 구현
- [ ] 2D Gaussian splatting 알고리즘 구현
- [ ] Alpha blending 및 sorting
- [ ] 최적화 (벡터화, CUDA kernel 고려)

**산출물**:
- 완성된 `GaussianRenderer2D` 클래스
- 단위 테스트

**검증**:
```python
# 단순 테스트: 단일 Gaussian splat
renderer = GaussianRenderer2D(256, 256)
params = torch.tensor([[
    128.0, 128.0,  # mean (center)
    2.0, 2.0,      # scale (log)
    0.0,           # rotation
    1.0, 0.0, 0.0, # color (red)
    2.0            # opacity (sigmoid)
]])
rgb, alpha = renderer.render(params, None, None)

# Check: red blob at center
assert rgb[128, 128, 0] > 0.5
assert alpha[128, 128] > 0.5
```

### Phase 4: Config 통합 (GPU 필요, 2-3시간)

**작업**:
- [ ] Config에 `gaussian_mode`, `gaussian_config` 추가
- [ ] `train_script.py` 수정
- [ ] 모든 스크립트에서 새 인터페이스 지원
- [ ] 2D/3D 각각 학습 테스트

**산출물**:
- `configs/markerless_mouse_nerf_2d.json`
- 수정된 학습/평가 스크립트

**검증**:
```bash
# 2D 모드 debug 학습
python train_script.py \
  --config configs/markerless_mouse_nerf_2d_debug.json \
  --epochs 10

# 3D 모드 기존 방식 (regression test)
python train_script.py \
  --config configs/markerless_mouse_nerf_3d_debug.json \
  --epochs 10
```

### Phase 5: 성능 비교 실험 (GPU 필요, 5-10시간)

**작업**:
- [ ] 2D 학습 (100 epochs)
- [ ] 3D 학습 (100 epochs)
- [ ] 메트릭 비교 (IOU, PSNR, SSIM, L1)
- [ ] 속도 비교 (FPS, training time)
- [ ] 메모리 사용량 비교

**산출물**:
- 비교 실험 보고서
- 시각화 (metrics plots, rendered images)

**검증**:
```bash
# 결과 분석
python analyze_results.py \
  --config1 configs/markerless_mouse_nerf_2d.json \
  --config2 configs/markerless_mouse_nerf_3d.json \
  --output docs/reports/2d_vs_3d_comparison.pdf
```

---

## 6. 예상 결과

### 6.1 2D Gaussian Splatting 특성

**예상 장점**:
- **속도**: 3D보다 1.5-2배 빠른 렌더링
- **메모리**: 파라미터 수 35% 감소 (9 vs 14)
- **안정성**: Depth collapse 문제 없음

**예상 단점**:
- **Multi-view consistency**: View 간 일관성 낮음
- **Novel view synthesis**: 새로운 각도에서 품질 저하
- **3D 정보**: 깊이 정보 손실

### 6.2 Use Case 별 추천

| Use Case | 추천 모드 | 이유 |
|----------|----------|------|
| Pose tracking (single view) | 2D | 빠르고 안정적 |
| Novel view synthesis | 3D | Multi-view consistency 필요 |
| 3D reconstruction | 3D | 명확 |
| Behavior analysis (overhead) | 2D | 2D 충분 |
| Multi-camera tracking | 3D | View consistency |
| Real-time application | 2D | 속도 우선 |

### 6.3 Hybrid 전략 (Future Work)

**단계별 전환**:
1. **초기 학습**: 2D로 빠른 수렴
2. **Fine-tuning**: 3D로 multi-view refinement
3. **Inference**: 용도에 따라 선택

**구현 방안**:
```python
# Phase 1: 2D training (epoch 0-50)
model = PoseSplatter(..., gaussian_mode="2d")
train(model, epochs=50)

# Phase 2: 3D fine-tuning (epoch 51-100)
model.switch_to_3d()  # Convert 2D params to 3D
train(model, epochs=50)
```

---

## 7. Testing 전략

### 7.1 Unit Tests

```python
# tests/test_gaussian_renderer.py (NEW FILE)

import torch
import pytest
from src.gaussian_renderer import (
    GaussianRenderer2D,
    GaussianRenderer3D,
    create_renderer
)

class TestGaussianRenderer2D:
    def test_num_params(self):
        renderer = GaussianRenderer2D(256, 256)
        assert renderer.get_num_params() == 9

    def test_single_gaussian(self):
        renderer = GaussianRenderer2D(256, 256, device="cpu")
        params = torch.tensor([[
            128.0, 128.0,  # center
            1.0, 1.0,      # scale
            0.0,           # rotation
            1.0, 0.0, 0.0, # red
            2.0            # opacity (sigmoid)
        ]])
        rgb, alpha = renderer.render(params, None, None)

        assert rgb.shape == (256, 256, 3)
        assert alpha.shape == (256, 256)
        assert rgb[128, 128, 0] > 0.5  # Red at center

    def test_out_of_bounds(self):
        renderer = GaussianRenderer2D(256, 256, device="cpu")
        params = torch.tensor([[
            -100.0, -100.0,  # outside image
            1.0, 1.0,
            0.0,
            1.0, 0.0, 0.0,
            2.0
        ]])
        rgb, alpha = renderer.render(params, None, None)
        assert rgb.sum() == 0  # No contribution

class TestGaussianRenderer3D:
    def test_num_params(self):
        renderer = GaussianRenderer3D(256, 256)
        assert renderer.get_num_params() == 14

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_render(self):
        renderer = GaussianRenderer3D(256, 256, device="cuda")
        params = torch.randn(100, 14).cuda()

        viewmat = torch.eye(4).cuda()
        K = torch.tensor([
            [256, 0, 128],
            [0, 256, 128],
            [0, 0, 1]
        ], dtype=torch.float32).cuda()

        rgb, alpha = renderer.render(params, viewmat, K)
        assert rgb.shape == (256, 256, 3)
        assert alpha.shape == (256, 256)

class TestRendererFactory:
    def test_create_2d(self):
        renderer = create_renderer("2d", 256, 256, device="cpu")
        assert isinstance(renderer, GaussianRenderer2D)

    def test_create_3d(self):
        renderer = create_renderer("3d", 256, 256, device="cpu")
        assert isinstance(renderer, GaussianRenderer3D)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            create_renderer("invalid", 256, 256)
```

### 7.2 Integration Tests

```bash
# tests/test_integration.sh

#!/bin/bash
set -e

echo "Testing 2D renderer integration..."
python train_script.py \
  --config configs/test_2d_integration.json \
  --epochs 1 \
  --max_batches 5

echo "Testing 3D renderer integration..."
python train_script.py \
  --config configs/test_3d_integration.json \
  --epochs 1 \
  --max_batches 5

echo "Testing renderer switching..."
python tests/test_renderer_switch.py

echo "All integration tests passed!"
```

### 7.3 Performance Tests

```python
# tests/benchmark_renderers.py

import torch
import time
from src.gaussian_renderer import create_renderer

def benchmark_renderer(mode, num_gaussians=1000, num_iters=100):
    renderer = create_renderer(mode, 256, 256, device="cuda")
    params = torch.randn(num_gaussians, renderer.get_num_params()).cuda()

    viewmat = torch.eye(4).cuda()
    K = torch.tensor([[256, 0, 128], [0, 256, 128], [0, 0, 1]]).float().cuda()

    # Warmup
    for _ in range(10):
        rgb, alpha = renderer.render(params, viewmat, K)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        rgb, alpha = renderer.render(params, viewmat, K)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    fps = num_iters / elapsed
    return fps

if __name__ == "__main__":
    fps_2d = benchmark_renderer("2d")
    fps_3d = benchmark_renderer("3d")

    print(f"2D Renderer: {fps_2d:.1f} FPS")
    print(f"3D Renderer: {fps_3d:.1f} FPS")
    print(f"Speedup: {fps_2d / fps_3d:.2f}x")
```

---

## 8. 문서화 계획

### 8.1 API 문서

**생성 도구**: Sphinx
**포함 내용**:
- `GaussianRenderer` interface
- `GaussianRenderer2D` implementation details
- `GaussianRenderer3D` implementation details
- `create_renderer()` factory function
- Parameter layout specification

### 8.2 사용자 가이드

**파일**: `docs/guides/gaussian_renderer_usage.md`
**포함 내용**:
- Quick start
- Config 설정 방법
- 2D vs 3D 선택 가이드
- 예제 코드
- Troubleshooting

### 8.3 개발자 가이드

**파일**: `docs/guides/extending_renderers.md`
**포함 내용**:
- 새로운 renderer 추가 방법
- 테스트 작성 가이드
- 최적화 팁
- CUDA kernel 작성 (advanced)

---

## 9. 리스크 및 대응

### 9.1 기술적 리스크

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|----------|
| 2D 렌더링 속도 느림 | 중 | 높음 | CUDA kernel 최적화 |
| 3D 리팩토링 시 regression | 중 | 높음 | 철저한 테스트, checkpoint 비교 |
| 메모리 오버헤드 | 낮음 | 중 | Profiling 후 최적화 |
| 2D 품질 저하 | 높음 | 중 | Use case 명확화, hybrid 접근 |

### 9.2 일정 리스크

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|----------|
| 2D 구현 복잡도 과소평가 | 중 | 중 | MVP 우선, 고급 기능 later |
| CUDA 디버깅 시간 초과 | 중 | 높음 | CPU fallback, 점진적 최적화 |
| 성능 비교 실험 장시간 | 낮음 | 낮음 | Debug config 우선 |

---

## 10. Success Criteria

### 10.1 기능 요구사항

- [x] 2D/3D renderer 추상화 인터페이스 정의
- [ ] 3D renderer 리팩토링 완료 (기존 결과와 동일)
- [ ] 2D renderer 구현 완료
- [ ] Config 기반 전환 가능
- [ ] 모든 단위 테스트 통과
- [ ] 통합 테스트 통과

### 10.2 성능 요구사항

- [ ] 2D renderer 최소 30 FPS @ 256x256 (1000 Gaussians)
- [ ] 3D renderer 기존 성능 유지 (regression 없음)
- [ ] 메모리 오버헤드 < 10%

### 10.3 품질 요구사항

- [ ] 2D IOU > 0.7 (단일 뷰 기준)
- [ ] 3D IOU > 0.8 (기존 대비 유지)
- [ ] 코드 coverage > 80%
- [ ] Docstring 완비

---

## 11. Timeline

```
Week 1: Design & Interface
├─ Day 1-2: Phase 1 완료 (인터페이스 정의)
└─ Day 3: 설계 리뷰

Week 2: 3D Refactoring
├─ Day 1-2: Phase 2 완료 (3D 리팩토링)
└─ Day 3: Regression test

Week 3: 2D Implementation
├─ Day 1-3: Phase 3 완료 (2D 구현)
└─ Day 4-5: 단위 테스트

Week 4: Integration
├─ Day 1-2: Phase 4 완료 (Config 통합)
└─ Day 3-5: 통합 테스트

Week 5: Experiments
├─ Day 1-3: Phase 5 (성능 비교)
└─ Day 4-5: 문서화 및 보고서

Total: 5 weeks (part-time, 약 60-80 시간)
```

---

## 12. 참고 자료

**논문**:
- "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)
- "2D Gaussian Splatting for Geometrically Accurate Radiance Fields" (Huang et al., 2024)

**코드베이스**:
- gsplat: https://github.com/nerfstudio-project/gsplat
- diff-gaussian-rasterization: https://github.com/graphdeco-inria/diff-gaussian-rasterization

**관련 프로젝트**:
- `/home/joon/dev/pose-splatter/src/model.py` (현재 3D 구현)
- `/home/joon/dev/pose-splatter/train_script.py` (학습 루프)

---

**작성자**: Claude Code
**작성일**: 2025-11-12
**버전**: 1.0
**상태**: Design Proposal
