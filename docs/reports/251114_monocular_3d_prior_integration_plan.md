# Monocular 3D Prior Integration Plan

**날짜**: 2025-11-14
**프로젝트**: pose-splatter
**목적**: 6-view → 1-view (monocular) 전환 및 3D prior 통합

---

## Executive Summary

### 현재 상황

**Pose-Splatter (현재)**:
- ✅ 6-view multi-camera setup 필요
- ✅ 3D Gaussian Splatting (volume carving 기반)
- ✅ 2D/3D GS renderer 구현 완료 (Phase 1)
- ❌ Monocular input 불가능 (6 cameras 필수)

**사용자 요구사항**:
1. **Monocular input**: 1개 카메라로 작동
2. **3D prior 활용**: Mouse mesh/shape prior로 depth ambiguity 해결
3. **통합 접근**: 3DAnimals (Fauna) + MAMMAL mouse 활용

**참조 프로젝트**:
- `/home/joon/dev/3DAnimals`: Fauna 4D model (학습 기반)
- `/home/joon/dev/MAMMAL_mouse`: MAMMAL parametric model (fitting 기반)

---

## 1. 문제 분석

### 1.1 Pose-Splatter의 Multi-View 의존성

**Volume Carving 단계**:
```python
# src/shape_carver.py
class ShapeCarver:
    def __init__(self, ..., intrinsics, extrinsics):
        # 6 cameras required for shape carving
        self.intrinsics = intrinsics  # [6, 3, 3]
        self.extrinsics = extrinsics  # [6, 4, 4]

    def __call__(self, masks, images, ...):
        # Carve 3D volume from multi-view silhouettes
        volume = self.carve_volume_from_masks(masks)
        return volume
```

**문제점**:
- Multi-view consistency 필수
- Single view에서는 depth 정보 없음
- Volume carving 불가능

### 1.2 3D Prior의 필요성

**Depth Ambiguity**:
```
Single RGB image → Infinite 3D interpretations
Need: Shape prior to constrain solution space
```

**해결 방안**:
1. **Parametric Model** (MAMMAL): Template mesh + articulation
2. **Learned Model** (Fauna): Neural implicit representation
3. **Hybrid**: Parametric prior + Neural refinement

---

## 2. 참조 프로젝트 분석

### 2.1 MAMMAL Mouse

**위치**: `/home/joon/dev/MAMMAL_mouse`

**핵심 기능**:
```python
# articulation_th.py
class ArticulationTorch:
    def __init__(self):
        # Template mesh: ~1000 vertices
        self.v_template_th  # [V, 3] vertices
        self.faces_th       # [F, 3] faces

        # Articulation: 22 joints
        self.jointnum = 22
        self.kintree_table  # Parent-child relationships

    def forward(self, thetas, bone_lengths, R, T, s):
        """
        Args:
            thetas: Joint angles [B, 22, 3]
            bone_lengths: Bone parameters [B, 28]
            R: Global rotation [B, 3] (axis-angle)
            T: Global translation [B, 3]
            s: Global scale [B, 1]

        Returns:
            vertices: [B, V, 3] posed mesh vertices
        """
        # Skinning with LBS (Linear Blend Skinning)
        vertices = self.articulate(thetas, bone_lengths)
        vertices = s * vertices @ R + T
        return vertices
```

**Monocular Fitting Pipeline** (`fit_monocular.py`):
```python
class MonocularMAMMALFitter:
    def fit(self, rgb, mask):
        # 1. Extract 2D keypoints from mask
        keypoints_2d = self.extract_keypoints_from_mask(mask)

        # 2. Initialize pose from keypoints
        thetas, bone_lengths, R, T, s = self.initialize_pose(keypoints_2d)

        # 3. Optimize parameters
        for iter in range(max_iters):
            # Project 3D mesh to 2D
            mesh_3d = self.model(thetas, bone_lengths, R, T, s)
            mesh_2d = self.project(mesh_3d, camera_params)

            # Compute losses
            loss_silhouette = chamfer_distance(mesh_2d, mask_contour)
            loss_keypoints = mse(mesh_2d_joints, keypoints_2d)
            loss_reg = regularization(thetas, bone_lengths)

            loss = loss_silhouette + loss_keypoints + loss_reg
            loss.backward()
            optimizer.step()

        return mesh_3d
```

**장점**:
- ✅ Interpretable parameters (joint angles, bone lengths)
- ✅ Fast fitting (seconds per frame)
- ✅ Physically plausible results
- ✅ No training data needed

**단점**:
- ❌ Template mesh 고정 (topology 제한)
- ❌ Fine details 부족
- ❌ Texture 정보 없음

### 2.2 3DAnimals (Fauna)

**위치**: `/home/joon/dev/3DAnimals`

**핵심 구조**:
```python
# model/models/Fauna4D.py
class Fauna4D(AnimalModel):
    def __init__(self, ..., enable_deform=True):
        # SDF field (implicit representation)
        self.sdf_network = SDFNetwork(...)

        # Articulation network
        self.articulation_network = ArticulationNetwork(...)

        # Appearance (texture)
        self.texture_network = TextureNetwork(...)

    def forward(self, rgb, mask, dino_features):
        """
        Args:
            rgb: [B, 3, H, W]
            mask: [B, 1, H, W]
            dino_features: [B, 384, H/16, W/16] (from ViT)

        Returns:
            sdf: Signed distance field
            mesh: Extracted mesh (Marching cubes)
            articulation: Joint angles
        """
        # 1. Feature encoding
        features = self.encoder(rgb, mask, dino_features)

        # 2. SDF prediction
        sdf = self.sdf_network(features, query_points)

        # 3. Articulation
        joint_angles = self.articulation_network(features)

        # 4. Mesh extraction
        mesh = self.extract_mesh(sdf)  # Marching cubes

        return sdf, mesh, joint_angles
```

**학습 방식**:
- Multi-view 이미지 사용
- Self-supervised (silhouette, optical flow)
- Category-specific (각 동물 종마다 학습 필요)

**Mouse Scale 문제** (FAUNA_MOUSE_EXECUTION_PLAN.md 참고):
```
Fauna는 mouse-scale animals에 근본적 한계:
- Grid resolution: 64^3 → 128^3로 증가해도 crash
- Spatial extent: 작은 동물은 grid에 담기 어려움
- 실험 결과: 3-7 iterations에서 반복적 crash
```

**장점**:
- ✅ High-quality reconstruction
- ✅ Implicit representation (arbitrary topology)
- ✅ Learned from data (realistic shapes)

**단점**:
- ❌ Mouse-scale 지원 불가 (확인됨)
- ❌ Training data 필요 (large dataset)
- ❌ Slow inference (SDF sampling + marching cubes)
- ❌ DINO features 추출 필요

---

## 3. 통합 전략

### 3.1 제안하는 Hybrid 접근법

**Pipeline Overview**:
```
Monocular RGB + Mask
       ↓
[Stage 1] MAMMAL Fitting (Initial 3D Prior)
       ↓
[Stage 2] Volume Initialization (MAMMAL → Volume)
       ↓
[Stage 3] Gaussian Splatting Refinement
       ↓
Final Reconstruction
```

### 3.2 Stage 1: MAMMAL Fitting (3D Prior Extraction)

**목적**: Monocular image → Coarse 3D mesh

**구현**:
```python
# pose-splatter/src/mammal_prior.py (NEW FILE)

from MAMMAL_mouse.articulation_th import ArticulationTorch
from MAMMAL_mouse.fit_monocular import MonocularMAMMALFitter

class MAMMALPriorExtractor:
    """
    Extract 3D prior from MAMMAL mouse model
    """

    def __init__(self, device='cuda'):
        self.fitter = MonocularMAMMALFitter(device=device)
        self.model = self.fitter.model

    def extract_3d_prior(self, rgb, mask):
        """
        Args:
            rgb: [H, W, 3] RGB image
            mask: [H, W] binary mask

        Returns:
            vertices: [V, 3] mesh vertices
            faces: [F, 3] mesh faces
            params: Dict of MAMMAL parameters
        """
        # Fit MAMMAL model
        result = self.fitter.fit(rgb, mask, max_iters=100)

        vertices = result['vertices']  # [V, 3]
        faces = self.model.faces_th    # [F, 3]
        params = {
            'thetas': result['thetas'],          # Joint angles
            'bone_lengths': result['bone_lengths'],
            'R': result['R'],                    # Global rotation
            'T': result['T'],                    # Global translation
            's': result['s'],                    # Global scale
        }

        return vertices, faces, params
```

**예상 시간**: ~5-10 seconds per frame

### 3.3 Stage 2: Volume Initialization

**목적**: MAMMAL mesh → Voxel volume (Pose-Splatter 입력)

**방법 1: Mesh Rasterization**
```python
# pose-splatter/src/mesh_to_volume.py (NEW FILE)

import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import rasterize_meshes

def mammal_mesh_to_volume(
    vertices,     # [V, 3]
    faces,        # [F, 3]
    grid_size=112,
    volume_idx=[[0, 96], [16, 96], [25, 89]],
    ell=0.22
):
    """
    Convert MAMMAL mesh to volumetric representation

    Returns:
        volume: [C, D, H, W] voxel grid
    """
    # 1. Create 3D grid
    grid = create_3d_grid(ell, grid_size, volume_idx)  # [N, 3]

    # 2. Compute SDF (Signed Distance Field)
    mesh = Meshes(verts=[vertices], faces=[faces])
    sdf = compute_sdf(grid, mesh)  # [N]

    # 3. Convert SDF to occupancy
    occupancy = torch.sigmoid(-sdf / 0.01)  # [N]

    # 4. Reshape to volume
    volume = occupancy.reshape(
        volume_idx[2][1] - volume_idx[2][0],  # D
        volume_idx[1][1] - volume_idx[1][0],  # H
        volume_idx[0][1] - volume_idx[0][0],  # W
    )

    # 5. Add channel dimension (RGBA)
    volume = volume.unsqueeze(0).repeat(4, 1, 1, 1)  # [4, D, H, W]

    return volume
```

**방법 2: Differentiable Rasterization**
```python
def mammal_mesh_to_multiview_volume(
    vertices, faces,
    viewpoints=6,  # Simulate multi-view
    ...
):
    """
    Render mesh from multiple viewpoints and carve volume
    (Similar to original multi-view setup but using rendered views)
    """
    # Generate synthetic views from mesh
    views = []
    for angle in np.linspace(0, 2*np.pi, viewpoints):
        view = render_mesh(vertices, faces, camera_angle=angle)
        views.append(view)

    # Use existing shape carver with synthetic views
    carver = ShapeCarver(...)
    volume = carver(views['mask'], views['rgb'], ...)

    return volume
```

### 3.4 Stage 3: Gaussian Splatting Refinement

**기존 Pose-Splatter Pipeline 사용**:
```python
# pose-splatter/train_script.py (MODIFIED)

# Initialize volume from MAMMAL prior
if config.use_mammal_prior:
    prior_extractor = MAMMALPriorExtractor(device=device)

    # Extract 3D prior (only once or per few frames)
    vertices, faces, params = prior_extractor.extract_3d_prior(
        rgb=dataset[0]['rgb'],
        mask=dataset[0]['mask']
    )

    # Convert to volume
    initial_volume = mammal_mesh_to_volume(vertices, faces, ...)

    # Use as initialization for model
    model.carver.set_initial_volume(initial_volume)

# Normal training
for epoch in range(epochs):
    for batch in dataloader:
        # Volume processing with MAMMAL prior as initialization
        volume = model.carver(mask, img, p_3d, angle, initial=initial_volume)

        # UNet refinement
        volume = model.process_volume(volume)

        # Gaussian rendering (2D or 3D)
        rgb, alpha = model(...)

        # Loss computation
        loss = compute_loss(rgb, alpha, target_rgb, target_mask)
        loss.backward()
        optimizer.step()
```

---

## 4. 구현 단계별 계획

### Phase 1: MAMMAL Integration (비-GPU 일부, GPU 필요)

**예상 시간**: 4-6시간

**Step 1.1: MAMMAL 코드 통합** (2시간)
- [ ] MAMMAL_mouse 코드를 pose-splatter로 복사
- [ ] Dependencies 확인 및 설치
- [ ] Import paths 수정
- [ ] Test script 작성

**Step 1.2: Prior Extractor 구현** (2-3시간)
- [ ] `src/mammal_prior.py` 작성
- [ ] Monocular fitting wrapper
- [ ] Mesh extraction 인터페이스
- [ ] Unit tests

**Step 1.3: 단일 이미지 테스트** (1시간, GPU)
- [ ] Sample image로 MAMMAL fitting 테스트
- [ ] 결과 시각화 (mesh overlay)
- [ ] Parameters validation

**산출물**:
- `src/mammal_prior.py`
- `tests/test_mammal_prior.py`
- Sample mesh output

### Phase 2: Volume Conversion (GPU 필요)

**예상 시간**: 3-4시간

**Step 2.1: Mesh-to-Volume 구현** (2시간)
- [ ] `src/mesh_to_volume.py` 작성
- [ ] SDF computation (pytorch3d or custom)
- [ ] Grid sampling
- [ ] Voxelization

**Step 2.2: Synthetic Multi-View (Optional)** (2시간)
- [ ] Differentiable renderer setup
- [ ] Generate 6 views from mesh
- [ ] Feed to existing shape carver

**Step 2.3: Volume Quality 검증** (1시간)
- [ ] Visualize voxel grid
- [ ] Compare with original multi-view volume
- [ ] Adjust parameters

**산출물**:
- `src/mesh_to_volume.py`
- Voxel visualization scripts

### Phase 3: Pipeline Integration (GPU 필요)

**예상 시간**: 4-5시간

**Step 3.1: Config 확장** (1시간)
- [ ] Add `use_mammal_prior` flag
- [ ] Add `mammal_config` section
- [ ] Update config loading logic

**Step 3.2: Training Loop 수정** (2시간)
- [ ] Integrate prior extraction
- [ ] Initialize volume from mesh
- [ ] Update forward pass

**Step 3.3: Monocular Dataset** (1시간)
- [ ] Create monocular dataloader
- [ ] Single-view sampling
- [ ] Test data loading

**Step 3.4: End-to-End Test** (1-2시간, GPU)
- [ ] Train debug mode (10 epochs)
- [ ] Check convergence
- [ ] Visualize results

**산출물**:
- Updated `train_script.py`
- Monocular config file
- Training logs

### Phase 4: Evaluation & Comparison (GPU 필요)

**예상 시간**: 5-8시간

**Step 4.1: Baseline Experiments** (3시간)
- [ ] Multi-view (6 cameras) - 기존 방식
- [ ] Monocular + MAMMAL prior
- [ ] Monocular without prior (baseline)

**Step 4.2: Metrics** (2시간)
- [ ] Chamfer distance (mesh reconstruction)
- [ ] Silhouette IoU
- [ ] PSNR, SSIM (rendering quality)

**Step 4.3: Visualization** (2-3시간)
- [ ] Side-by-side comparisons
- [ ] Novel view synthesis
- [ ] Mesh quality plots

**산출물**:
- Comparison report
- Visualization videos
- Metrics plots

### Phase 5: 2D/3D GS Testing (GPU 필요)

**예상 시간**: 3-4시간

**Step 5.1: 3D GS Validation** (1-2시간)
- [ ] Load existing checkpoint
- [ ] Run inference
- [ ] Pixel-wise regression test

**Step 5.2: 2D GS Training** (2시간)
- [ ] Train with 2D renderer
- [ ] Compare speed vs 3D
- [ ] Quality comparison

**산출물**:
- 2D/3D comparison report

---

## 5. 예상 결과 및 Trade-offs

### 5.1 MAMMAL Prior 효과

**With MAMMAL Prior**:
- ✅ Depth ambiguity 해결
- ✅ Physically plausible shapes
- ✅ Consistent across frames
- ⚠️ Coarse initial guess (fine details 부족)

**Without MAMMAL Prior** (Monocular only):
- ❌ Depth collapse (flat reconstruction)
- ❌ Ambiguous solutions
- ❌ Unstable across frames

### 5.2 Multi-View vs Monocular

| 측면 | Multi-View (6 cam) | Monocular + Prior |
|------|-------------------|-------------------|
| Setup | ❌ Complex | ✅ Simple |
| Accuracy | ✅ High | ⚠️ Medium |
| Speed | ⚠️ 6x data | ✅ 1x data |
| Prior 필요 | ❌ No | ✅ Yes |
| Generalization | ✅ Good | ⚠️ Template-dependent |

### 5.3 2D vs 3D GS (Monocular 시나리오)

**3D GS**:
- ✅ Novel view synthesis
- ❌ Requires good depth estimate
- Monocular에서 MAMMAL prior 필수

**2D GS**:
- ✅ Direct 2D optimization (depth 불필요)
- ✅ Faster convergence
- ❌ Poor novel view synthesis
- Monocular에서 더 적합할 수 있음

---

## 6. Alternative: Fauna 재시도 (선택적)

### 6.1 Fauna Mouse 문제 재검토

**2025-11-12 실험 결과**:
- Mouse-scale에서 반복적 crash
- Grid resolution 증가해도 실패

**재시도 조건** (매우 낮은 우선순위):
1. DINO features 추출 필요 (1-2시간)
2. Debug mode 실행 (15-20분)
3. 예상 결과: 여전히 실패

**결론**: Fauna는 현재 mouse에 부적합, MAMMAL 우선

---

## 7. 최종 권장 로드맵

### 즉시 실행 (GPU 준비 완료)

**Priority 1: 2D/3D GS Validation** (2-3시간)
```bash
# 1. 기존 3D checkpoint 테스트
python tests/test_model_integration.py

# 2. 기존 checkpoint 로드 및 inference
python render_image.py \
  --config configs/markerless_mouse_nerf_extended.json \
  --checkpoint output/markerless_mouse_nerf_extended/checkpoint.pt

# 3. 2D debug 학습
python train_script.py \
  --config configs/markerless_mouse_nerf_2d_test.json \
  --epochs 10
```

**Priority 2: MAMMAL Integration** (4-6시간)
```bash
# 1. MAMMAL prior 구현
# Create src/mammal_prior.py
# Create src/mesh_to_volume.py

# 2. Single image test
python tests/test_mammal_prior.py

# 3. Volume conversion test
python tests/test_mesh_to_volume.py
```

**Priority 3: Monocular Pipeline** (4-5시간)
```bash
# 1. Update train_script.py
# Add MAMMAL prior initialization

# 2. Create monocular config
# configs/markerless_mouse_nerf_monocular.json

# 3. Debug training
python train_script.py \
  --config configs/markerless_mouse_nerf_monocular.json \
  --epochs 10
```

**Priority 4: Evaluation** (5-8시간)
```bash
# Compare:
# - Multi-view baseline
# - Monocular + MAMMAL
# - 2D vs 3D GS
```

**총 예상 시간**: 15-22시간

---

## 8. Config 예시

### 8.1 Monocular + MAMMAL Prior Config

```json
{
    "data_directory": "/home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf/",
    "project_directory": "/home/joon/dev/pose-splatter/output/markerless_mouse_monocular/",

    "video_fns": [
        "videos_undist/0.mp4"  // Only 1 view!
    ],
    "mask_video_fns": [
        "simpleclick_undist/0.mp4"
    ],

    "holdout_views": [],  // No holdout for monocular
    "image_width": 1152,
    "image_height": 1024,
    "image_downsample": 4,

    "use_mammal_prior": true,  // NEW
    "mammal_config": {         // NEW
        "checkpoint_path": "/home/joon/dev/MAMMAL_mouse/checkpoints/mammal_mouse.pth",
        "fitting_iterations": 100,
        "optimize_bone_lengths": true,
        "optimize_articulation": true
    },

    "gaussian_mode": "3d",  // or "2d" for simpler reconstruction
    "gaussian_config": {},

    "grid_size": 112,
    "ell": 0.22,
    "volume_idx": [[0, 96], [16, 96], [25, 89]],

    "lr": 1e-4,
    "valid_every": 10,
    "plot_every": 5,
    "save_every": 5
}
```

---

## 9. Success Criteria

### 9.1 Phase 1 (MAMMAL Integration)
- [ ] MAMMAL fitting works on single image
- [ ] Mesh visualization looks reasonable
- [ ] Parameters are physically plausible

### 9.2 Phase 2 (Volume Conversion)
- [ ] Voxel grid captures mesh shape
- [ ] Volume compatible with Pose-Splatter input
- [ ] Visual inspection shows correct structure

### 9.3 Phase 3 (Pipeline Integration)
- [ ] Training runs without errors
- [ ] Loss decreases over epochs
- [ ] Reconstructed mesh resembles input

### 9.4 Phase 4 (Evaluation)
- [ ] Monocular + Prior > Monocular only
- [ ] Quality acceptable compared to multi-view
- [ ] Consistent across frames

### 9.5 Phase 5 (2D/3D GS)
- [ ] 2D/3D both work with monocular input
- [ ] Speed/quality trade-off understood
- [ ] Recommendations for use cases

---

## 10. Risks & Mitigation

| Risk | 확률 | 영향 | 대응 |
|------|------|------|------|
| MAMMAL fitting 실패 | 중 | 높음 | Keypoint detection 개선, manual initialization |
| Volume conversion 품질 저하 | 중 | 중 | SDF resolution 증가, smoothing |
| Training divergence | 중 | 높음 | Learning rate 조정, prior weight tuning |
| 2D GS 느린 속도 | 높음 | 중 | CUDA kernel 최적화, 또는 3D 사용 |
| Monocular depth ambiguity | 높음 | 높음 | Prior weight 증가, multi-frame consistency |

---

## 11. 다음 단계

**Immediate (오늘/내일)**:
1. ✅ 2D/3D GS validation (통합 테스트 실행)
2. ✅ 기존 checkpoint 검증
3. ⏳ MAMMAL prior 구현 시작

**Short-term (이번 주)**:
1. MAMMAL integration 완료
2. Monocular pipeline 구축
3. Debug training

**Medium-term (다음 주)**:
1. Full training (100 epochs)
2. Evaluation & comparison
3. Documentation

---

**작성자**: Claude Code
**작성일**: 2025-11-14
**상태**: Design Phase - Ready for Implementation
**다음 작업**: Phase 1 - 2D/3D GS Validation
