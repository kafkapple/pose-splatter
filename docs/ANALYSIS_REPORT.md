# 📊 Pose Splatter 프로젝트 종합 분석 보고서

**작성일**: 2025-11-08
**버전**: 1.0
**프로젝트**: Pose Splatter (NeurIPS 2025)

---

## 목차
1. [연구 배경 및 목적](#1-연구-배경-및-목적)
2. [핵심 모델 파이프라인](#2-핵심-모델-파이프라인)
3. [데이터 입출력 구조](#3-데이터-입출력-구조)
4. [실행 파이프라인](#4-실행-파이프라인)
5. [학습 과정](#5-학습-과정)
6. [필요 환경 설정](#6-필요-환경-설정)
7. [누락된 기능 분석](#7-누락된-기능-분석)
8. [우선순위별 실행 계획](#8-우선순위별-실행-계획)
9. [즉시 실행 가능한 작업](#9-즉시-실행-가능한-작업)
10. [트러블슈팅 가이드](#10-트러블슈팅-가이드)

---

## 1. 연구 배경 및 목적

### 1.1 논문 정보
- **제목**: Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance
- **저자**: Jack Goffinet*, Youngjo Min*, Carlo Tomasi, David E. Carlson (* equal contribution)
- **출판**: NeurIPS 2025
- **arXiv**: https://arxiv.org/abs/2505.18342

### 1.2 핵심 혁신
- **Shape Carving + 3D Gaussian Splatting** 기반 동물 자세 추정 프레임워크
- 수동 주석 및 프레임별 최적화 불필요
- 동물 기하학 사전 지식 없이 작동
- 회전 불변 시각 임베딩 제공
- 쥐, 쥐, 호핑새 등 다양한 동물에 적용 가능

### 1.3 기대 효과
> "대규모, 종단적 행동 분석을 가능하게 하여 유전형, 신경 활동, 미세 행동을 전례 없는 해상도로 매핑"

---

## 2. 핵심 모델 파이프라인

### 2.1 전체 아키텍처

```
Multi-view Images + Silhouettes
    ↓
┌──────────────────────────────┐
│  Shape Carving Module        │ → 3D Volume [4, n1, n2, n3]
│  (src/shape_carver.py)       │   - Channel 0: Occupancy
└──────────────────────────────┘   - Channel 1-3: RGB colors
    ↓
┌──────────────────────────────┐
│  3D U-Net Stack (× 3)        │ → Refined Volume [8, n1, n2, n3]
│  (src/unet_3d.py)            │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│  Gaussian Parameter Network  │ → Per-voxel Gaussian params
│  (MLP: 8→128→14)             │   - Quats (4D), Scales (3D)
└──────────────────────────────┘   - Opacities (1D), Colors (3D)
    ↓                                - Delta means (3D)
┌──────────────────────────────┐
│  3D Gaussian Splatting       │ → Rendered Image [H, W, 3]
│  (gsplat.rendering)          │   + Alpha mask [H, W, 1]
└──────────────────────────────┘
```

### 2.2 주요 컴포넌트 상세

#### A. Shape Carving Module (`src/shape_carver.py:ShapeCarver`)

**목적**: 다중 시점 실루엣 이미지로부터 3D volume 생성

**입력**:
- `mask`: [C, 1, H, W] - 각 카메라 시점의 실루엣 마스크
- `rgb`: [C, 3, H, W] - 각 카메라 시점의 RGB 이미지
- `center`: [3] - 3D 공간의 중심 좌표
- `angle`: scalar - z축 회전 각도

**출력**:
- `volume`: [4, n1, n2, n3] - 4채널 3D volume

**핵심 알고리즘**:
1. **Grid 생성 및 변환** (`create_3d_grid`):
   - 균일한 3D voxel grid 생성
   - 회전(angle)과 이동(center) 적용

2. **투영 및 샘플링** (`project_points_torch`):
   - 각 voxel을 모든 카메라에 투영
   - 카메라 내부/외부 파라미터 사용

3. **가시성 결정** (`ray_cast_visibility_torch`):
   - Ray casting으로 occluded voxel 판별
   - torch_scatter로 효율적 구현

4. **색상 계산** (`compute_voxel_colors_torch`):
   - 가시적 카메라의 가중 평균
   - 비가시 카메라는 낮은 가중치 부여

**코드 위치**: `src/shape_carver.py:308-381`

---

#### B. 3D U-Net (`src/unet_3d.py:Unet3D`)

**목적**: Volume feature 추출 및 refinement

**아키텍처**:
```
Input [4, n1, n2, n3]
  ↓
Encoder (5 levels)
  - Conv3d + BatchNorm + LeakyReLU (×2) per level
  - MaxPool3d for downsampling
  - Base filters: 8 → 16 → 32 → 64 → 128
  ↓
Bottleneck MLP
  - Flatten → Linear(128×n_prod, 512) → ReLU → Linear(512, 512)
  ↓
Decoder (4 levels)
  - ConvTranspose3d for upsampling
  - Skip connections from encoder
  - Conv3d + BatchNorm + LeakyReLU (×2) per level
  ↓
Final Conv: 8 → out_channels (default: 8)
Output [8, n1, n2, n3]
```

**특징**:
- **Skip connections**: U-Net 구조로 detail 보존
- **Identity initialization** (`init_unet_primary_skip`):
  - 초기에 입력을 거의 그대로 통과
  - 안정적인 학습 시작
- **Residual design**: 입력 채널을 출력에 직접 복사

**하이퍼파라미터**:
- `base_filters`: 8 (기본값)
- `z_dim`: 512 (latent dimension)
- `input_size`: [80, 80, 48] (예시, volume_idx에 따라 변함)

**코드 위치**: `src/unet_3d.py:75-168`

---

#### C. Gaussian Parameter Network

**목적**: Volume feature → 3D Gaussian 파라미터 변환

**구조**:
```python
nn.Sequential(
    nn.Linear(out_channels, 128),  # out_channels = 8
    nn.ReLU(),
    nn.Linear(128, 14),           # 14 = 4+3+1+3+3
)
```

**출력 파라미터** (14-dim per voxel):
- **Quaternions** (4D): Gaussian의 회전 (정규화 필요)
- **Scales** (3D): 각 축 방향 크기 (exp 적용 후 사용)
- **Opacities** (1D): 불투명도 [0, 1]
- **Colors** (3D): RGB 색상 (sigmoid → [0, 1])
- **Delta means** (3D): voxel 중심에서 미세 조정 (tanh 사용)

**후처리**:
```python
colors = sigmoid(colors).clamp(0.0, 0.99)
scales = exp(scales + scale_offset)  # scale_offset ≈ -5.5
opacities = sigmoid((probs[mask] - threshold) / (1 - threshold))
means = grid_centers[mask] + 2 * voxel_size * tanh(delta_means)
```

**코드 위치**: `src/model.py:89-94, 167-200`

---

#### D. 3D Gaussian Splatting

**목적**: Gaussian primitives를 2D 이미지로 렌더링

**라이브러리**: `gsplat.rendering.rasterization`
- 미분 가능한 rasterization
- 효율적인 GPU 구현
- Real-time rendering 최적화

**렌더링 과정**:
1. Gaussians를 카메라 좌표계로 변환
2. 2D로 투영 (covariance matrix 계산)
3. Tile-based rasterization
4. Alpha blending (앞에서 뒤로 정렬)

**출력**:
```python
rgb = render + (1 - alpha) * background_color  # [1, H, W, 3]
alpha = [1, H, W, 1]
```

**코드 위치**: `src/model.py:220-246`

---

## 3. 데이터 입출력 구조

### 3.1 원본 데이터 형식

**비디오 데이터**:
```
data_directory/
├── Camera1/
│   └── 0.mp4           # RGB video
├── Camera2/
│   └── 0.mp4
├── ...
├── Camera6/
│   └── 0.mp4
└── mask_videos/
    ├── 1.mp4           # Silhouette masks (grayscale)
    ├── 2.mp4
    ├── ...
    └── 6.mp4
```

**카메라 캘리브레이션** (`camera_params_*.h5`):
```
HDF5 structure:
/camera_parameters/
  ├── rotation: [C, 3, 3]      # 회전 행렬
  ├── translation: [C, 3]      # 이동 벡터
  └── intrinsic: [C, 3, 3]     # 내부 파라미터 (fx, fy, cx, cy)
```

**Config 파일 예시** (`configs/mouse_4.json`):
```json
{
  "data_directory": "/path/to/data/mouse/",
  "project_directory": "/path/to/project/mouse_4_cameras/",
  "mask_video_fns": ["mask_videos/1.mp4", ..., "mask_videos/6.mp4"],
  "video_fns": ["Camera1/0.mp4", ..., "Camera6/0.mp4"],
  "holdout_views": [5, 1],
  "image_width": 2048,
  "image_height": 1536,
  "image_downsample": 4,
  "fps": 30,
  "frame_jump": 5,
  "ell": 0.22,
  "grid_size": 112,
  "volume_idx": [[0, 96], [16, 96], [25, 89]],
  "lr": 1e-4,
  "img_lambda": 0.5,
  "ssim_lambda": 0.0
}
```

### 3.2 전처리된 데이터 형식

**이미지 데이터** (`images.h5` → `images.zarr`):
```
Shape: [N_frames, C_cameras, H, W, 3]
Dtype: uint8
Compression: gzip (level 2)
Storage: Zarr (for efficient random access)
```

**자세 데이터** (`center_rotation.npz`):
```python
{
  "centers": [N_frames, 3],    # 3D center coordinates
  "angles": [N_frames],         # Z-axis rotation angles
}
```

### 3.3 데이터 로더 (`src/data.py:FrameDataset`)

**출력 형식**:
```python
mask:     torch.Tensor [C, H, W]          # Binary silhouettes
img:      torch.Tensor [C, 3, H, W]       # RGB images (normalized to [0,1])
p_3d:     torch.Tensor [3]                # 3D center
angle:    float                           # Rotation angle (radians)
view_idx: int                             # Camera index to render
```

**데이터 분할**:
- **Train**: frames 0 ~ N/3
- **Valid**: frames N/3 ~ 2N/3
- **Test**: frames 2N/3 ~ N

**코드 위치**: `src/data.py:15-77`

---

## 4. 실행 파이프라인

### 4.1 전처리 단계 (Steps 1-5)

```bash
# Step 1: 카메라 Up Direction 추정
python estimate_up_direction.py config.json
# 출력: vertical_lines.npz (up vector)

# Step 2: 각 프레임의 중심 및 회전 계산
python calculate_center_rotation.py config.json
# 출력: center_rotation.npz (centers, angles)

# Step 3: Volume crop 인덱스 결정
python calculate_crop_indices.py config.json
# 출력: volume_sum.npy
# 콘솔에 volume_idx 출력 → config.json에 수동 입력

# Step 4: 이미지를 HDF5로 저장
python write_images.py config.json
# 출력: images/images.h5 [N, C, H, W, 3]

# Step 5: HDF5 → Zarr 변환
python copy_to_zarr.py images/images.h5 images/images.zarr
# 출력: images/images.zarr (학습 시 사용)
```

### 4.2 학습 단계 (Step 6)

```bash
# 기본 학습
python train_script.py config.json --epochs 50

# Ablation 실험 (U-Net 없이)
python train_script.py config.json --epochs 50 --ablation

# 중단된 학습 재개
python train_script.py config.json --load --epochs 100

# 디버그 모드 (빠른 검증)
python train_script.py config.json --epochs 5 --max_batches 50
```

**출력**:
- `project_directory/reconstruction.pdf`: 예측 이미지
- `project_directory/loss.pdf`: 학습 곡선
- `project_directory/checkpoint.pt`: 모델 체크포인트

### 4.3 평가 및 추론 단계 (Steps 7-10)

```bash
# Step 7: 정량적 평가
python evaluate_model.py config.json
# 출력: rendered_images.h5, metrics_test.csv

# Step 8: 단일 이미지 렌더링
python render_image.py config.json <frame_num> <view_num>
# 예: python render_image.py config.json 100 0
# 출력: renders/render_100_0_0.0_0.0_0.0_0.0.png

# Step 8 (고급): 자세 변형
python render_image.py config.json 100 0 \
    --angle_offset 0.5 \
    --delta_x 0.1 --delta_y 0.0 --delta_z 0.05

# Step 9: Visual features 계산
python calculate_visual_features.py config.json
# 출력: features.npy

# Step 10: Visual embedding 계산
python calculate_visual_embedding.py config.json
# 출력: embedding.npy
```

---

## 5. 학습 과정

### 5.1 손실 함수

**총 손실**:
```python
total_loss = iou_loss + ssim_loss + img_loss
```

**1. IoU Loss (실루엣 매칭)**:
```python
def get_iou_loss(pred_alpha, target_mask, eps=1e-6):
    intersection = (pred_alpha * target_mask).sum()
    union = (pred_alpha + target_mask - pred_alpha * target_mask).sum()
    iou = (intersection + eps) / (union + eps)
    return 1 - iou
```
- 목적: 렌더링된 alpha mask와 GT silhouette 일치
- 범위: [0, 1]

**2. SSIM Loss (구조적 유사도)**:
```python
ssim_loss = ssim_lambda * (1.0 - SSIM(pred_img, target_img))
```
- 기본적으로 비활성화 (`ssim_lambda = 0.0`)
- 활성화 시 이미지 구조 보존

**3. Image Loss (L1 픽셀 차이)**:
```python
img_loss = img_lambda * torch.abs(target_img - pred_img).sum() / mask.sum()
```
- 마스크 영역 내에서만 계산
- `img_lambda = 0.5` (기본값)

### 5.2 학습 알고리즘

**Optimizer**: Adam
- Learning rate: `1e-4` (기본값)
- No learning rate scheduling

**학습 루프**:
```python
for epoch in range(n_epochs):
    for mask, img, p_3d, angle, view_idx in train_loader:
        # Forward
        rgb, alpha = model(mask, img, p_3d, angle, view_idx)

        # Loss
        loss = iou_loss + img_loss + ssim_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation (every valid_every epochs)
    if epoch % valid_every == 0:
        val_loss = calculate_validation_loss(...)

    # Visualization (every plot_every epochs)
    if epoch % plot_every == 0:
        plot_predictions(...)
        plot_losses(...)

    # Checkpoint (every save_every epochs)
    if epoch % save_every == 0:
        torch.save({...}, checkpoint_fn)
```

### 5.3 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `lr` | 1e-4 | Learning rate |
| `img_lambda` | 0.5 | Image loss weight |
| `ssim_lambda` | 0.0 | SSIM loss weight (보통 0) |
| `valid_every` | 5 | Validation 주기 (epochs) |
| `plot_every` | 1 | Visualization 주기 |
| `save_every` | 1 | Checkpoint 저장 주기 |
| `image_downsample` | 4 | 이미지 해상도 감소 비율 |
| `ell` | 0.22 | Volume 크기 (m) |
| `grid_size` | 112 | Voxel 해상도 |
| `min_n` | 1024 | 최소 Gaussian 개수 |
| `max_n` | 16000 | 최대 Gaussian 개수 |
| `num_unets` | 3 | U-Net 개수 |

---

## 6. 필요 환경 설정

### 6.1 하드웨어 요구사항

**최소 사양**:
- GPU: NVIDIA GPU with CUDA support (8GB+ VRAM)
- RAM: 16GB+
- Storage: 50GB+ (데이터셋 크기에 따라)

**권장 사양**:
- GPU: NVIDIA RTX 3090 / A100 (24GB+ VRAM)
- RAM: 32GB+
- Storage: 100GB+ SSD

**CUDA 아키텍처**:
- 코드에 하드코딩: `os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"`
- Ampere (3000 series) 이상 권장
- 다른 GPU 사용 시 해당 라인 수정 필요

### 6.2 소프트웨어 의존성

**핵심 라이브러리**:
```
python >= 3.10
torch >= 2.0.0
pytorch-cuda = 11.8
gsplat                    # 3D Gaussian Splatting
torch-scatter             # Sparse scatter operations
zarr                      # Chunked array storage
h5py                      # HDF5 file format
opencv-python (cv2)       # Video processing
torchmetrics              # SSIM, PSNR 등
matplotlib                # Visualization
Pillow                    # Image I/O
tqdm                      # Progress bars
joblib                    # Parallel processing
```

**설치 방법 (Conda)**:
```bash
# 1. 환경 생성
conda create -n pose-splatter python=3.10 -y
conda activate pose-splatter

# 2. PyTorch (CUDA 11.8)
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. gsplat (source 빌드 필요할 수 있음)
pip install gsplat

# 4. torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 5. 기타 라이브러리
pip install zarr h5py opencv-python torchmetrics matplotlib Pillow tqdm joblib
```

### 6.3 환경 검증

```bash
# CUDA 확인
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# 라이브러리 import 테스트
python -c "import gsplat; import torch_scatter; import zarr; print('All imports OK')"

# 모델 로딩 테스트
python -c "from src.model import PoseSplatter; print('Model import OK')"
```

---

## 7. 누락된 기능 분석

### 7.1 README Checklist

```markdown
### Project Checklist
- [x] Code on GitHub
- [ ] Camera-ready on arXiv
- [ ] Add links to data          ← 데이터셋 다운로드 링크 없음
- [ ] Add more detailed usage    ← 상세 사용법 부족
```

### 7.2 누락된 파일 및 문서

**1. 환경 설정 파일**:
- ❌ `requirements.txt`
- ❌ `environment.yml`
- ❌ `setup.py` 또는 `pyproject.toml`

**2. 데이터셋 관련**:
- ❌ 예제 데이터셋 (작은 샘플이라도)
- ❌ 카메라 캘리브레이션 방법 문서
- ❌ 데이터 수집 가이드
- ❌ 데이터 포맷 명세

**3. 문서화**:
- ❌ API 문서 (Docstrings 부족)
- ❌ 하이퍼파라미터 가이드
- ❌ 트러블슈팅 가이드
- ⚠️ Config 파라미터 설명 (일부만 존재)

**4. 시각화 도구**:
- ⚠️ `plot_voxels.py` 존재하지만 사용법 불명확
- ❌ 3D volume viewer
- ❌ Gaussian primitives 시각화
- ❌ 학습 진행 실시간 모니터링

### 7.3 코드 개선 필요 사항

**1. 하드코딩된 값**:
```python
# train_script.py:23, evaluate_model.py:21, render_image.py:18
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"  # ← 특정 GPU 아키텍처 고정
```
→ Config로 이동 또는 자동 감지 필요

**2. 에러 핸들링 부족**:
- 파일 없을 때 명확한 에러 메시지 부족
- GPU 메모리 부족 시 대응 코드 없음
- Config 검증 로직 미흡

**3. 주석 및 문서화**:
- Docstring 거의 없음 (일부 파일만 존재)
- Type hints 부족
- 복잡한 알고리즘 설명 부족

**4. 확장성**:
- 카메라 개수 하드코딩 (C=6 가정)
- 다양한 동물 종에 대한 자동 설정 부족

### 7.4 구현 권장 사항

**우선순위: 높음**
1. ✅ `requirements.txt` 작성
2. ✅ 환경 설정 가이드 작성
3. ✅ Config 파라미터 설명 문서
4. ⚠️ 예제 데이터 또는 다운로드 링크 제공

**우선순위: 중간**
5. 3D 시각화 도구 개선
6. TensorBoard 통합
7. Checkpoint resume 로직 개선
8. 데이터 전처리 자동화

**우선순위: 낮음**
9. Docker 이미지 제공
10. Weights & Biases 통합
11. Multi-GPU 학습 지원
12. 자동 하이퍼파라미터 튜닝

---

## 8. 우선순위별 실행 계획

### Phase 1: 환경 구축 ⭐⭐⭐ (필수)

**소요 시간**: 30분 ~ 1시간

**단계**:
1. **Conda 환경 생성**:
   ```bash
   conda create -n pose-splatter python=3.10 -y
   conda activate pose-splatter
   ```

2. **PyTorch 설치** (CUDA 11.8):
   ```bash
   conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
   ```

3. **gsplat 설치**:
   ```bash
   pip install gsplat
   # 실패 시: pip install gsplat --no-cache-dir
   ```

4. **torch-scatter 설치**:
   ```bash
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   ```

5. **기타 라이브러리**:
   ```bash
   pip install zarr h5py opencv-python torchmetrics matplotlib Pillow tqdm joblib
   ```

6. **설치 검증**:
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   python -c "import gsplat, torch_scatter, zarr; print('All OK')"
   python -c "from src.model import PoseSplatter; print('Model OK')"
   ```

**체크리스트**:
- [ ] Conda 환경 생성 완료
- [ ] PyTorch CUDA 사용 가능
- [ ] gsplat 설치 완료
- [ ] torch-scatter 설치 완료
- [ ] 모든 의존성 import 성공

---

### Phase 2: 데이터 준비 ⭐⭐⭐ (필수)

**소요 시간**: 변동 (데이터 확보에 따라)

**현재 상황**: 공개 데이터셋 링크 없음 ❌

**옵션 A: 저자에게 데이터 요청**
```bash
# GitHub Issue 생성
# 제목: "[Data Request] Sample dataset for reproduction"
# 내용:
# - 작은 예제 데이터 요청 (1-2분 분량)
# - 또는 데이터 다운로드 링크 요청
```

**옵션 B: 자체 데이터 수집**

**필요 장비**:
- 최소 4개, 권장 6개 카메라
- 동기화 가능한 설정
- 균일한 조명

**데이터 수집 절차**:
1. **카메라 캘리브레이션**:
   - 체커보드 패턴 사용
   - OpenCV `calibrateCamera` 함수
   - 출력: 내부/외부 파라미터

2. **비디오 촬영**:
   - 동물 행동 녹화
   - 모든 카메라 동기화
   - 배경 제거 가능한 설정 권장

3. **실루엣 마스크 생성**:
   - 배경 차분 또는 딥러닝 세그멘테이션
   - 바이너리 마스크 (0 또는 255)

**옵션 C: 공개 Multi-view 데이터셋 활용**
- CMU Panoptic Dataset
- DeepLabCut 3D 데이터
- (카메라 캘리브레이션 포함 필요)

**체크리스트**:
- [ ] 원본 비디오 확보 (RGB × C)
- [ ] 실루엣 마스크 생성 완료
- [ ] 카메라 캘리브레이션 완료
- [ ] Config 파일 작성 (경로 수정)

---

### Phase 3: 전처리 파이프라인 ⭐⭐ (데이터 확보 후)

**소요 시간**: 1-3시간 (비디오 길이에 따라)

**전제 조건**: Phase 2 완료

**실행 순서**:

```bash
# 0. Config 파일 수정
# - data_directory: 비디오 경로
# - project_directory: 출력 경로
# - mask_video_fns, video_fns: 파일 이름

# 1. Up direction 추정 (1-5분)
python estimate_up_direction.py config.json
# 출력: vertical_lines.npz

# 2. Center & Rotation 계산 (5-15분)
python calculate_center_rotation.py config.json
# 출력: center_rotation.npz

# 3. Volume crop 인덱스 결정 (5-10분)
python calculate_crop_indices.py config.json
# 콘솔 출력에서 volume_idx 확인
# → config.json에 수동 입력

# 4. 이미지 HDF5 저장 (30분 - 2시간)
python write_images.py config.json
# 출력: images/images.h5
# 병렬 처리 (CPU 코어 수만큼)

# 5. ZARR 변환 (10-30분)
python copy_to_zarr.py images/images.h5 images/images.zarr
# 출력: images/images.zarr
```

**디버깅 팁**:
- 각 단계 출력 파일 크기 확인
- `volume_idx`가 이상하면 `calculate_crop_indices.py` 재실행
- 메모리 부족 시 `frame_jump` 증가

**체크리스트**:
- [ ] vertical_lines.npz 생성
- [ ] center_rotation.npz 생성
- [ ] volume_idx 확인 및 config 업데이트
- [ ] images.h5 생성 확인
- [ ] images.zarr 생성 확인

---

### Phase 4: 모델 학습 ⭐⭐ (전처리 완료 후)

**소요 시간**: 수 시간 ~ 수 일 (데이터 크기, GPU 성능에 따라)

**디버그 모드 (먼저 실행 권장)**:
```bash
# 작은 배치로 빠른 검증
python train_script.py config.json --epochs 5 --max_batches 50
# 소요: 5-10분
# 목적:
# - 데이터 로딩 정상 확인
# - forward/backward pass 성공 확인
# - 메모리 사용량 체크
```

**전체 학습**:
```bash
# 기본 학습 (50 epochs)
python train_script.py config.json --epochs 50
# 소요: 수 시간 (GPU, 데이터 크기에 따라)

# Ablation 실험
python train_script.py config.json --epochs 50 --ablation
# U-Net 없이 학습 (비교용)

# 중단된 학습 재개
python train_script.py config.json --load --epochs 100
```

**모니터링**:
```bash
# 주기적으로 확인
watch -n 60 "ls -lh project_directory/*.{pdf,pt}"

# GPU 사용량
watch -n 1 nvidia-smi
```

**출력 파일**:
- `reconstruction.pdf`: 예측 품질 (매 epoch)
- `loss.pdf`: 학습/검증 곡선
- `checkpoint.pt`: 모델 가중치

**중단 조건**:
- Validation loss가 수렴
- Train loss는 감소하지만 validation loss 증가 (overfitting)

**체크리스트**:
- [ ] 디버그 모드 성공
- [ ] 학습 loss 감소 확인
- [ ] Reconstruction 품질 개선 확인
- [ ] Checkpoint 정상 저장
- [ ] GPU 메모리 overflow 없음

---

### Phase 5: 평가 및 시각화 ⭐ (학습 완료 후)

**소요 시간**: 30분 ~ 2시간

**정량적 평가**:
```bash
# 모든 test frames 렌더링 및 메트릭 계산
python evaluate_model.py config.json
# 출력:
# - rendered_images.h5: [N_test, C, H, W, 4]
# - metrics_test.csv: IoU, SSIM, PSNR, L1

# CSV 확인
cat project_directory/metrics_test.csv
```

**정성적 평가 (이미지 렌더링)**:
```bash
# 단일 프레임, 단일 시점
python render_image.py config.json 100 0
# 출력: renders/render_100_0_0.0_0.0_0.0_0.0.png

# 다양한 각도
for angle in 0.0 0.5 1.0; do
  python render_image.py config.json 100 0 --angle_offset $angle
done

# 위치 변화
python render_image.py config.json 100 0 \
  --delta_x 0.1 --delta_y 0.0 --delta_z 0.05
```

**Novel view synthesis (holdout views)**:
```bash
# config에서 holdout_views = [5, 1]이면
python render_image.py config.json 100 5  # 카메라 5
python render_image.py config.json 100 1  # 카메라 1
# 이 시점들은 학습 중 보지 못함 → 일반화 성능 확인
```

**체크리스트**:
- [ ] 정량적 메트릭 확인 (IoU > 0.9 권장)
- [ ] 렌더링 이미지 시각적 품질 확인
- [ ] Novel view 성능 확인
- [ ] Ablation 모델과 비교 (U-Net 효과)

---

### Phase 6: 고급 기능 (선택)

**우선순위: 낮음**

**시각 특징 추출**:
```bash
# 모든 프레임의 latent features
python calculate_visual_features.py config.json
# 출력: features.npy

# Dimensionality reduction (UMAP 등)
python calculate_visual_embedding.py config.json
# 출력: embedding.npy
```

**사용 사례**:
- Behavior clustering
- Anomaly detection
- Trajectory analysis

---

## 9. 즉시 실행 가능한 작업

### 9.1 환경 준비 (데이터 없이 가능) ✅

```bash
# 전체 스크립트
conda create -n pose-splatter python=3.10 -y
conda activate pose-splatter
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install gsplat torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install zarr h5py opencv-python torchmetrics matplotlib Pillow tqdm joblib

# 검증
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from src.model import PoseSplatter; print('OK')"
```

### 9.2 코드 검증 (데이터 없이 가능) ✅

```bash
# 모델 구조 확인
python -c "from src.model import PoseSplatter; import torch; print(PoseSplatter)"

# U-Net 테스트
python src/unet_3d.py
# 출력: Initial MSE between input and first 4 output channels = ...

# Config 로딩 테스트
python -c "from src.config_utils import Config; c = Config('configs/mouse_4.json'); print('ell:', c.ell)"

# Shape carving 모듈 import
python -c "from src.shape_carver import ShapeCarver; print('OK')"
```

### 9.3 문서 작성 (지금 바로 가능) ✅

**1. requirements.txt 생성**:
```bash
# 환경 구축 후
pip freeze > requirements.txt

# 또는 수동 작성
cat > requirements.txt << EOF
torch==2.0.0
torchvision==0.15.0
gsplat
torch-scatter
zarr
h5py
opencv-python
torchmetrics
matplotlib
Pillow
tqdm
joblib
EOF
```

**2. environment.yml 생성**:
```yaml
name: pose-splatter
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.0.0
  - torchvision=0.15.0
  - pytorch-cuda=11.8
  - pip
  - pip:
    - gsplat
    - torch-scatter
    - zarr
    - h5py
    - opencv-python
    - torchmetrics
    - matplotlib
    - Pillow
    - tqdm
    - joblib
```

**3. 상세 사용 가이드 작성** (이 문서)

### 9.4 데이터 요청 (지금 바로 가능) ✅

```markdown
# GitHub Issue 템플릿

Title: [Data Request] Sample dataset for reproduction

Body:
Hi @jackgoffinet @youngjomin,

Thank you for open-sourcing this excellent work! I'm trying to reproduce
the results but couldn't find the dataset download links.

Could you please provide:
1. A small sample dataset (1-2 minutes) for testing the pipeline?
2. Or links to download the full datasets (rat, mouse, finch)?
3. Documentation on the camera calibration format (camera_params_*.h5)?

This would greatly help the community reproduce and build upon your work.

Thank you!
```

---

## 10. 트러블슈팅 가이드

### 10.1 설치 문제

#### 문제: `gsplat` 설치 실패
```
ERROR: Could not build wheels for gsplat
```

**해결**:
```bash
# CUDA 버전 확인
nvcc --version

# 캐시 삭제 후 재설치
pip cache purge
pip install gsplat --no-cache-dir

# source에서 빌드
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
pip install -e .
```

---

#### 문제: `torch-scatter` 설치 실패
```
ERROR: No matching distribution found for torch-scatter
```

**해결**:
```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# 해당 버전에 맞는 URL 사용
# PyTorch 2.0.0 + CUDA 11.8:
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 또는 conda 사용
conda install pytorch-scatter -c pyg
```

---

### 10.2 데이터 로딩 문제

#### 문제: `Zarr file does not exist`
```python
FileNotFoundError: Zarr file does not exist: /path/to/images.zarr
```

**원인**: HDF5 → Zarr 변환 누락

**해결**:
```bash
# copy_to_zarr.py 실행 확인
python copy_to_zarr.py images/images.h5 images/images.zarr

# 파일 존재 확인
ls -lh images/
# 기대: images.h5, images.zarr/ (디렉토리)
```

---

#### 문제: Config 경로 오류
```python
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data/...'
```

**원인**: Config에 절대 경로 또는 잘못된 상대 경로

**해결**:
```json
// config.json 수정
{
  "data_directory": "/absolute/path/to/data/mouse/",
  "project_directory": "/absolute/path/to/project/mouse_4_cameras/",
  ...
}
```

---

### 10.3 GPU 메모리 문제

#### 문제: `CUDA out of memory`
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.00 GiB (GPU 0; 7.79 GiB total capacity)
```

**해결 방법 (우선순위 순)**:

**1. 이미지 해상도 감소**:
```json
// config.json
"image_downsample": 8,  // 4 → 8
```

**2. Grid 해상도 감소**:
```json
"grid_size": 64,  // 112 → 64
```

**3. Gaussian 개수 제한**:
```json
"max_n": 8000,  // 16000 → 8000
```

**4. U-Net base filters 감소**:
```python
# train_script.py
model = PoseSplatter(
    ...
    base_filters=4,  # 8 → 4
)
```

**5. Ablation 모드 사용**:
```bash
python train_script.py config.json --ablation
# U-Net 없이 학습 → 메모리 절약
```

---

### 10.4 학습 문제

#### 문제: Loss가 감소하지 않음
```
Epoch 10: loss = 1.234 (no improvement)
```

**원인 및 해결**:

**1. Learning rate 너무 낮음**:
```json
"lr": 1e-3,  // 1e-4 → 1e-3
```

**2. Loss weights 불균형**:
```json
"img_lambda": 1.0,   // 0.5 → 1.0
"ssim_lambda": 0.1,  // 0.0 → 0.1
```

**3. 데이터 문제**:
```bash
# Reconstruction 이미지 확인
open project_directory/reconstruction.pdf
# 완전히 검정/하양이면 데이터 로딩 문제
```

---

#### 문제: Overfitting (Validation loss 증가)
```
Epoch 30: train_loss = 0.05, val_loss = 0.20 (increasing)
```

**해결**:

**1. Early stopping**:
```python
# Validation loss가 증가하면 학습 중단
```

**2. 데이터 증강** (코드 수정 필요):
```python
# FrameDataset.__getitem__에 추가
angle_offset = np.random.uniform(-0.1, 0.1)
center_offset = np.random.normal(0, 0.01, 3)
```

---

### 10.5 렌더링 문제

#### 문제: 렌더링 이미지가 검정색
```
render_100_0_0.0_0.0_0.0_0.0.png is all black
```

**원인**:
- Gaussian 개수가 0 (모든 voxel이 threshold 이하)
- 잘못된 camera parameters

**해결**:

**1. Threshold 확인**:
```python
# src/model.py:54-56
prob_threshold=0.25,        # 낮추기: 0.25 → 0.1
mask_threshold=0.25,
mask_threshold_delta=0.05,
```

**2. Volume 확인**:
```python
# render_image.py에 디버그 추가
print("Volume shape:", volume.shape)
print("Volume range:", volume.min(), volume.max())
print("Num Gaussians:", (probs > threshold).sum())
```

---

#### 문제: 렌더링이 너무 느림
```
Rendering 1 image takes 10 seconds
```

**원인**: 너무 많은 Gaussians

**해결**:
```json
// config.json
"max_n": 8000,  // 16000 → 8000
"prob_threshold": 0.3,  // 0.25 → 0.3 (더 적은 Gaussians)
```

---

### 10.6 기타 문제

#### 문제: `TORCH_CUDA_ARCH_LIST` 경고
```
Warning: CUDA arch list does not match
```

**해결**:
```python
# train_script.py, evaluate_model.py, render_image.py
# 주석 처리하거나 GPU에 맞게 수정
# os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"  # RTX 3090

# GPU 아키텍처 확인
nvidia-smi --query-gpu=compute_cap --format=csv
# 출력 예: 8.6 (Ampere), 7.5 (Turing), 8.9 (Ada)
```

---

#### 문제: Multi-processing 오류
```
RuntimeError: DataLoader worker (pid 12345) is killed by signal
```

**해결**:
```python
# train_script.py
num_workers = 0  # 또는 1, 2 (len(os.sched_getaffinity(0)) 대신)
loader_kwargs = dict(batch_size=1, shuffle=True, num_workers=0)
```

---

## 11. 요약 및 다음 단계

### 11.1 현재 상태

✅ **완료**:
- 코드 구조 분석
- 모델 파이프라인 이해
- 실행 절차 파악
- 환경 설정 방법 정리

❌ **미완성**:
- 공개 데이터셋 없음
- 예제 실행 불가
- 일부 문서 부족

### 11.2 즉시 실행 가능

**지금 바로** (데이터 없이):
1. ✅ Conda 환경 구축
2. ✅ 의존성 설치
3. ✅ 코드 import 테스트
4. ✅ Documentation 작성

**데이터 확보 후**:
5. ⏳ 전처리 파이프라인 실행
6. ⏳ 모델 학습
7. ⏳ 평가 및 시각화

### 11.3 Next Steps

**우선순위 1** (필수):
- [ ] 저자에게 데이터 요청 (GitHub Issue)
- [ ] 환경 구축 및 검증
- [ ] requirements.txt 생성

**우선순위 2** (데이터 확보 시):
- [ ] 전처리 파이프라인 실행
- [ ] 디버그 모드로 학습 테스트
- [ ] 전체 학습 실행

**우선순위 3** (학습 완료 시):
- [ ] 정량적 평가
- [ ] 시각화 및 분석
- [ ] Ablation study

### 11.4 참고 자료

- **논문**: https://arxiv.org/abs/2505.18342
- **GitHub**: https://github.com/[author]/pose-splatter (추정)
- **gsplat**: https://github.com/nerfstudio-project/gsplat
- **3D Gaussian Splatting**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

**작성자 노트**:
이 문서는 코드 분석 결과를 바탕으로 작성되었습니다. 실제 실행 시 예상치 못한 문제가 발생할 수 있으므로, 디버그 모드로 먼저 테스트하는 것을 강력히 권장합니다.

공개 데이터셋이 제공되는 대로 이 가이드를 업데이트하겠습니다.
