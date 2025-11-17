# Pose Splatter 설정 가이드 - 환경별, 모드별 완전 가이드

## 목차
1. [환경별 설정](#환경별-설정)
2. [2D vs 3D Gaussian Splatting](#2d-vs-3d-gaussian-splatting)
3. [주요 파라미터 설명](#주요-파라미터-설명)
4. [Config 파일 생성 방법](#config-파일-생성-방법)
5. [환경별 최적 설정](#환경별-최적-설정)
6. [실전 예제](#실전-예제)

---

## 환경별 설정

### GPU별 권장 설정

| GPU 모델 | VRAM | 권장 모드 | image_downsample | grid_size | max_frames |
|----------|------|-----------|------------------|-----------|------------|
| **RTX 3060** | 12GB | 3D (gsplat) | 4 | 112 | 3600 |
| **RTX 3090** | 24GB | 2D 또는 3D | 4 | 128 | 3600 |
| **A6000** | 48GB | 2D (최고 품질) | 2-4 | 128-256 | 전체 |
| **RTX 4090** | 24GB | 2D 또는 3D | 2-4 | 128 | 3600 |
| **A100** | 40-80GB | 2D (최고 품질) | 2 | 256 | 전체 |

### 서버별 설정

#### **gpu05 (RTX 3060 12GB)**

```json
{
    "data_directory": "data/markerless_mouse_1_nerf/",
    "project_directory": "output/markerless_mouse_nerf_3d/",
    "image_downsample": 4,
    "grid_size": 112,
    "frame_jump": 5,
    "max_frames": 3600,
    "gaussian_mode": "3d",
    "gaussian_config": {}
}
```

**특징**:
- 3D GS (gsplat) 사용 → 메모리 효율적
- 중간 해상도 (288×256 pixels)
- 안정적인 학습

#### **bori (A6000 24GB 이상)**

```json
{
    "data_directory": "data/markerless_mouse_1_nerf/",
    "project_directory": "output/markerless_mouse_nerf_2d_high/",
    "image_downsample": 2,
    "grid_size": 128,
    "frame_jump": 5,
    "max_frames": 3600,
    "gaussian_mode": "2d",
    "gaussian_config": {
        "sigma_cutoff": 3.0,
        "kernel_size": 5,
        "batch_size": 5
    }
}
```

**특징**:
- 2D GS (custom) → 최고 품질
- 고해상도 (576×512 pixels)
- 느리지만 정밀한 재구성

---

## 2D vs 3D Gaussian Splatting

### **차이점 비교**

| 특징 | 2D Gaussian Splatting | 3D Gaussian Splatting (gsplat) |
|------|----------------------|-------------------------------|
| **렌더링** | 2D 이미지 평면에 투영 | 3D 공간에서 직접 렌더링 |
| **GPU 메모리** | 높음 (15-20GB) | 낮음 (4-8GB) |
| **학습 속도** | 느림 (8-12s/batch) | 빠름 (16-20s/epoch) |
| **이미지 품질** | 매우 높음 (~30 dB PSNR) | 높음 (~27 dB PSNR) |
| **구현** | 커스텀 구현 | gsplat 라이브러리 |
| **GPU 요구사항** | 24GB+ | 12GB+ |
| **사용 사례** | 연구, 최고 품질 | 프로덕션, 효율 |

### **1. 3D Gaussian Splatting (gsplat) - 기본**

#### **Config 설정**

```json
{
    "gaussian_mode": "3d",
    "gaussian_config": {}
}
```

#### **특징**

**장점**:
- ✅ 메모리 효율적 (RTX 3060 12GB에서 작동)
- ✅ 빠른 학습 속도
- ✅ 안정적인 수렴
- ✅ gsplat 라이브러리 사용 (검증됨)

**단점**:
- ❌ 2D보다 약간 낮은 품질
- ❌ 세밀한 디테일 부족할 수 있음

#### **언제 사용?**

- RTX 3060, RTX 3070 등 중간 GPU
- 빠른 프로토타이핑
- 실시간 렌더링 필요 시
- 안정성 우선

#### **메모리 사용량**

```python
# 예상 GPU 메모리 (RTX 3060 기준)
image_downsample = 4  # 288×256
grid_size = 112
batch_size = 1

# 메모리 사용:
# - Model: ~2GB
# - Images: ~1GB
# - Gradients: ~2GB
# - CUDA overhead: ~1GB
# Total: ~6GB (12GB 중)
```

---

### **2. 2D Gaussian Splatting (Custom) - 고품질**

#### **Config 설정**

```json
{
    "gaussian_mode": "2d",
    "gaussian_config": {
        "sigma_cutoff": 3.0,
        "kernel_size": 5,
        "batch_size": 5
    }
}
```

#### **파라미터 설명**

| 파라미터 | 의미 | 기본값 | 범위 | 효과 |
|----------|------|--------|------|------|
| `sigma_cutoff` | Gaussian 잘라내기 임계값 | 3.0 | 2.0-4.0 | 높을수록 더 넓은 영역 렌더링 |
| `kernel_size` | 커널 크기 (픽셀) | 5 | 3, 5, 7 | 홀수, 클수록 블러 |
| `batch_size` | Gaussian 배치 크기 | 5 | 1-10 | 클수록 빠르지만 메모리 많이 사용 |

#### **특징**

**장점**:
- ✅ 최고 이미지 품질 (PSNR ~30dB)
- ✅ 세밀한 디테일 재현
- ✅ 더 정밀한 포즈 추정

**단점**:
- ❌ 높은 메모리 요구 (24GB+)
- ❌ 느린 학습 속도
- ❌ 불안정할 수 있음

#### **언제 사용?**

- A6000, RTX 4090, A100 등 고사양 GPU
- 최고 품질 재구성 필요
- 연구 목적, 논문 작성
- 시간 여유 있을 때

#### **메모리 사용량**

```python
# 예상 GPU 메모리 (A6000 기준)
image_downsample = 2  # 576×512
grid_size = 128
batch_size = 5

# 메모리 사용:
# - Model: ~3GB
# - Images: ~3GB
# - Gaussians (batch=5): ~8GB
# - Gradients: ~4GB
# Total: ~18GB (24GB 중)
```

---

## 주요 파라미터 설명

### **1. 데이터 관련**

#### `data_directory`
```json
"data_directory": "data/markerless_mouse_1_nerf/"
```
- **의미**: 원본 데이터 위치
- **타입**: 상대 또는 절대 경로
- **권장**: 상대 경로 사용 (서버 간 이동 편리)

#### `project_directory`
```json
"project_directory": "output/markerless_mouse_nerf/"
```
- **의미**: 출력 저장 위치 (checkpoint, renders, logs)
- **타입**: 상대 또는 절대 경로
- **주의**: 실험마다 다른 디렉토리 사용

#### `holdout_views`
```json
"holdout_views": [5, 1]
```
- **의미**: 테스트용으로 제외할 카메라 인덱스
- **범위**: 0-5 (6개 카메라)
- **권장**: 2개 카메라 (validation, test 각 1개)

---

### **2. 이미지 해상도**

#### `image_downsample`
```json
"image_downsample": 4
```
- **의미**: 원본 이미지 해상도를 얼마나 줄일지
- **계산**: `new_size = original_size / downsample`

| downsample | 해상도 | GPU 메모리 | 품질 | 속도 |
|------------|--------|-----------|------|------|
| 2 | 576×512 | 매우 높음 | 최고 | 느림 |
| 4 | 288×256 | 보통 | 좋음 | 빠름 |
| 6 | 192×171 | 낮음 | 괜찮음 | 매우 빠름 |
| 8 | 144×128 | 매우 낮음 | 나쁨 | 초고속 |

**선택 기준**:
```
RTX 3060 (12GB) → downsample = 4
RTX 3090 (24GB) → downsample = 2-4
A6000 (48GB)    → downsample = 2
```

#### `grid_size`
```json
"grid_size": 112
```
- **의미**: 3D 볼륨 해상도 (N×N×N 그리드)
- **메모리**: `grid_size^3 × 8 bytes`

| grid_size | 복셀 수 | 메모리 | 품질 |
|-----------|---------|--------|------|
| 64 | 262K | 2MB | 낮음 |
| 96 | 884K | 7MB | 보통 |
| 112 | 1.4M | 11MB | 좋음 |
| 128 | 2.1M | 16MB | 매우 좋음 |
| 256 | 16.8M | 134MB | 최고 |

**선택 기준**:
```
빠른 테스트   → grid_size = 64-96
일반 학습     → grid_size = 112-128
최고 품질     → grid_size = 256 (A100만)
```

---

### **3. 학습 프레임**

#### `frame_jump`
```json
"frame_jump": 5
```
- **의미**: 몇 프레임마다 하나씩 샘플링
- **계산**: `num_frames = total_frames / frame_jump`

| frame_jump | 프레임 수 | 데이터 크기 | 학습 시간 |
|------------|-----------|-------------|----------|
| 1 | 18,000 | 거대 | 매우 느림 |
| 2 | 9,000 | 큼 | 느림 |
| 5 | 3,600 | 보통 | 적당 |
| 10 | 1,800 | 작음 | 빠름 |

**권장**:
```
전체 학습     → frame_jump = 5
빠른 테스트   → frame_jump = 10-20
최고 품질     → frame_jump = 2
```

#### `max_frames`
```json
"max_frames": 100
```
- **의미**: 최대 사용할 프레임 수 (디버그용)
- **사용**: 빠른 테스트 시에만 설정
- **주의**: 실제 학습 시 제거 또는 큰 값 설정

---

### **4. 학습 하이퍼파라미터**

#### `lr` (Learning Rate)
```json
"lr": 1e-4
```
- **범위**: 1e-5 ~ 1e-3
- **기본**: 1e-4 (0.0001)

| lr | 수렴 속도 | 안정성 | 최종 품질 |
|----|----------|--------|----------|
| 1e-3 | 매우 빠름 | 불안정 | 나쁨 |
| 1e-4 | 빠름 | 안정 | 좋음 ✅ |
| 1e-5 | 느림 | 매우 안정 | 최고 |

#### `img_lambda`
```json
"img_lambda": 0.5
```
- **의미**: RGB 이미지 loss 가중치
- **범위**: 0.1 ~ 1.0
- **기본**: 0.5

#### `ssim_lambda`
```json
"ssim_lambda": 0.0
```
- **의미**: SSIM loss 가중치 (구조적 유사도)
- **범위**: 0.0 ~ 0.5
- **권장**:
  - 0.0: L1 loss만 (빠름, 기본)
  - 0.1-0.2: SSIM 추가 (품질 향상, 느림)

---

### **5. 체크포인트 및 로깅**

#### `valid_every`
```json
"valid_every": 5
```
- **의미**: 몇 epoch마다 validation 수행
- **권장**: 5-10 epochs

#### `plot_every`
```json
"plot_every": 1
```
- **의미**: 몇 epoch마다 이미지 저장
- **권장**:
  - 디버그: 1 (매 epoch)
  - 학습: 5 (5 epochs마다)

#### `save_every`
```json
"save_every": 5
```
- **의미**: 몇 epoch마다 checkpoint 저장
- **권장**: 5-10 epochs
- **주의**: 너무 자주 저장하면 디스크 부족

---

## Config 파일 생성 방법

### **방법 1: 기존 Config 복사 후 수정**

```bash
# Baseline config 복사
cp configs/baseline/markerless_mouse_nerf.json \
   configs/experiments/my_experiment.json

# 에디터로 수정
vim configs/experiments/my_experiment.json
```

### **방법 2: Python 스크립트로 생성**

```python
import json

config = {
    "data_directory": "data/markerless_mouse_1_nerf/",
    "project_directory": "output/my_experiment/",

    # 비디오 파일들 (변경 불필요)
    "mask_video_fns": [f"simpleclick_undist/{i}.mp4" for i in range(6)],
    "video_fns": [f"videos_undist/{i}.mp4" for i in range(6)],

    "holdout_views": [5, 1],

    # 해상도 설정 (여기 수정!)
    "image_downsample": 4,
    "grid_size": 112,
    "frame_jump": 5,

    # Gaussian 모드 (여기 수정!)
    "gaussian_mode": "3d",  # "2d" 또는 "3d"
    "gaussian_config": {},  # 2D일 경우 추가 파라미터

    # 학습 설정 (여기 수정!)
    "lr": 1e-4,
    "img_lambda": 0.5,
    "ssim_lambda": 0.0,

    "valid_every": 5,
    "plot_every": 1,
    "save_every": 5,

    # 기타 (변경 불필요)
    "volume_directory": "volumes",
    "image_directory": "images",
    "render_directory": "renders",
    "image_compression_level": 2,
    "volume_compression_level": 2,
    "camera_fn": "camera_params.h5",
    "vertical_lines_fn": "vertical_lines.npz",
    "center_rotation_fn": "center_rotation.npz",
    "volume_sum_fn": "volume_sum.npy",
    "model_fn": "checkpoint.pt",
    "feature_fn": "features.npy",
    "embedding_fn": "embedding.npy",
    "image_width": 1152,
    "image_height": 1024,
    "adaptive_camera": False,
    "fps": 100,
    "ell": 0.22,
    "ell_tracking": 0.25,
    "volume_idx": [[0, 96], [16, 96], [25, 89]],
    "volume_fill_color": 0.38
}

# 저장
with open('configs/experiments/my_experiment.json', 'w') as f:
    json.dump(config, f, indent=4)
```

---

## 환경별 최적 설정

### **gpu05 (RTX 3060 12GB) - 3D GS**

```json
{
    "data_directory": "data/markerless_mouse_1_nerf/",
    "project_directory": "output/rtx3060_baseline/",
    "image_downsample": 4,
    "grid_size": 112,
    "frame_jump": 5,
    "gaussian_mode": "3d",
    "gaussian_config": {},
    "lr": 1e-4,
    "img_lambda": 0.5,
    "ssim_lambda": 0.0
}
```

**예상 성능**:
- 학습 시간: ~6-8 시간 (50 epochs)
- GPU 메모리: ~6-8GB
- PSNR: ~25-27 dB
- 속도: ~20s/epoch

---

### **bori (A6000 24GB) - 2D GS High Quality**

```json
{
    "data_directory": "data/markerless_mouse_1_nerf/",
    "project_directory": "output/a6000_high_quality/",
    "image_downsample": 2,
    "grid_size": 128,
    "frame_jump": 5,
    "gaussian_mode": "2d",
    "gaussian_config": {
        "sigma_cutoff": 3.0,
        "kernel_size": 5,
        "batch_size": 5
    },
    "lr": 1e-4,
    "img_lambda": 0.5,
    "ssim_lambda": 0.1
}
```

**예상 성능**:
- 학습 시간: ~10-15 시간 (50 epochs)
- GPU 메모리: ~18-20GB
- PSNR: ~28-30 dB
- 속도: ~10s/batch

---

### **A100 80GB - 2D GS Ultra Quality**

```json
{
    "data_directory": "data/markerless_mouse_1_nerf/",
    "project_directory": "output/a100_ultra/",
    "image_downsample": 1,
    "grid_size": 256,
    "frame_jump": 2,
    "gaussian_mode": "2d",
    "gaussian_config": {
        "sigma_cutoff": 4.0,
        "kernel_size": 7,
        "batch_size": 10
    },
    "lr": 5e-5,
    "img_lambda": 0.5,
    "ssim_lambda": 0.2
}
```

**예상 성능**:
- 학습 시간: ~20-30 시간 (50 epochs)
- GPU 메모리: ~60-70GB
- PSNR: ~32+ dB
- 최고 품질

---

## 실전 예제

### **예제 1: 빠른 디버그 (모든 GPU)**

```json
{
    "data_directory": "data/markerless_mouse_1_nerf/",
    "project_directory": "output/debug_test/",
    "image_downsample": 6,
    "grid_size": 64,
    "frame_jump": 10,
    "max_frames": 100,
    "gaussian_mode": "3d",
    "gaussian_config": {},
    "valid_every": 2,
    "plot_every": 1,
    "save_every": 5
}
```

**실행**:
```bash
bash scripts/training/run_training.sh \
  configs/experiments/debug_test.json --epochs 5
```

**소요 시간**: ~5-10분

---

### **예제 2: RTX 3060 - 프로덕션**

```bash
# 1. Config 생성
cat > configs/experiments/rtx3060_prod.json << 'EOF'
{
    "data_directory": "data/markerless_mouse_1_nerf/",
    "project_directory": "output/rtx3060_prod/",
    "mask_video_fns": ["simpleclick_undist/0.mp4", "simpleclick_undist/1.mp4",
                       "simpleclick_undist/2.mp4", "simpleclick_undist/3.mp4",
                       "simpleclick_undist/4.mp4", "simpleclick_undist/5.mp4"],
    "video_fns": ["videos_undist/0.mp4", "videos_undist/1.mp4",
                  "videos_undist/2.mp4", "videos_undist/3.mp4",
                  "videos_undist/4.mp4", "videos_undist/5.mp4"],
    "holdout_views": [5, 1],
    "image_downsample": 4,
    "grid_size": 112,
    "frame_jump": 5,
    "gaussian_mode": "3d",
    "gaussian_config": {},
    "lr": 1e-4,
    "img_lambda": 0.5,
    "ssim_lambda": 0.0,
    "valid_every": 5,
    "plot_every": 5,
    "save_every": 10,
    "volume_directory": "volumes",
    "image_directory": "images",
    "render_directory": "renders",
    "image_compression_level": 2,
    "volume_compression_level": 2,
    "camera_fn": "camera_params.h5",
    "vertical_lines_fn": "vertical_lines.npz",
    "center_rotation_fn": "center_rotation.npz",
    "volume_sum_fn": "volume_sum.npy",
    "model_fn": "checkpoint.pt",
    "feature_fn": "features.npy",
    "embedding_fn": "embedding.npy",
    "image_width": 1152,
    "image_height": 1024,
    "adaptive_camera": false,
    "fps": 100,
    "ell": 0.22,
    "ell_tracking": 0.25,
    "volume_idx": [[0, 96], [16, 96], [25, 89]],
    "volume_fill_color": 0.38
}
EOF

# 2. 학습 실행
bash scripts/training/run_training.sh \
  configs/experiments/rtx3060_prod.json --epochs 50
```

---

### **예제 3: A6000 - 2D High Quality**

```bash
# Config는 위의 "bori (A6000 24GB)" 설정 사용

# 학습 실행 (백그라운드)
nohup bash scripts/training/run_training.sh \
  configs/experiments/a6000_high_quality.json --epochs 50 \
  > training_a6000.log 2>&1 &

# 모니터링
tail -f training_a6000.log
watch -n 2 nvidia-smi
```

---

## 요약표

| 환경 | GPU | Mode | downsample | grid | 학습 시간 | PSNR |
|------|-----|------|------------|------|----------|------|
| **gpu05** | RTX 3060 | 3D | 4 | 112 | 6-8h | 25-27 |
| **bori** | A6000 | 2D | 2 | 128 | 10-15h | 28-30 |
| **A100** | A100 80GB | 2D | 1 | 256 | 20-30h | 32+ |
| **Debug** | Any | 3D | 6 | 64 | 5-10m | N/A |

**핵심**:
- RTX 3060: 3D GS, downsample=4, grid=112
- A6000/4090: 2D GS, downsample=2, grid=128
- A100: 2D GS, downsample=1, grid=256
