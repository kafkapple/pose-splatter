# Pose Splatter 전처리 파이프라인 - 초보자 가이드

## 목차
1. [개요](#개요)
2. [카메라 파라미터란?](#카메라-파라미터란)
3. [전처리 단계별 설명](#전처리-단계별-설명)
4. [실전 예제](#실전-예제)
5. [문제 해결](#문제-해결)

---

## 개요

Pose Splatter는 **다중 카메라 비디오**에서 동물의 3D 포즈와 외형을 재구성합니다.

### 입력 데이터
- **6개 카메라의 RGB 비디오** (`videos_undist/*.mp4`)
- **6개 카메라의 세그멘테이션 마스크** (`simpleclick_undist/*.mp4`)
- **카메라 캘리브레이션 정보** (`new_cam.pkl` → `camera_params.h5`)
- **2D 키포인트 검출 결과** (`keypoints2d_undist/*.pkl`)

### 출력 데이터
- **정규화된 3D 볼륨 데이터** (HDF5, ZARR 형식)
- **학습에 필요한 메타데이터** (중심점, 회전 행렬, 크롭 범위)

---

## 카메라 파라미터란?

### 1. 카메라 파라미터의 출처

**MAMMAL_mouse 데이터셋**에 이미 포함되어 있습니다!

```bash
/home/joon/dev/MAMMAL_mouse/data/examples/markerless_mouse_1_nerf/
└── new_cam.pkl  # 54MB - 카메라 캘리브레이션 정보
```

이 파일은 **DANNCE 프로젝트**에서 미리 계산된 것입니다:
- 체커보드 패턴을 6개 카메라로 촬영
- OpenCV의 카메라 캘리브레이션 알고리즘 사용
- 내부 파라미터(intrinsic), 외부 파라미터(extrinsic) 계산

### 2. 카메라 파라미터 구성 요소

#### **Intrinsic Matrix (K)** - 카메라 내부 파라미터 [3x3]
```
K = [ fx  0   cx ]
    [ 0   fy  cy ]
    [ 0   0   1  ]

fx, fy: 초점 거리 (픽셀 단위)
cx, cy: 주점 (principal point) - 이미지 중심
```

**의미**: 3D 점을 2D 이미지로 투영하는 방법
- `fx`, `fy`: 렌즈의 줌 정도
- `cx`, `cy`: 카메라 센서의 중심 위치

#### **Rotation Matrix (R)** - 카메라 회전 [3x3]
```
R = [ r11  r12  r13 ]
    [ r21  r22  r23 ]
    [ r31  r32  r33 ]
```

**의미**: 월드 좌표계 → 카메라 좌표계 회전 변환
- 각 카메라가 어느 방향을 바라보는지

#### **Translation Vector (T)** - 카메라 위치 [3x1]
```
T = [ tx ]
    [ ty ]
    [ tz ]
```

**의미**: 월드 좌표계에서 카메라의 위치 (미터 단위)

### 3. 카메라 파라미터 변환 과정

#### Step 0-1: Pickle → HDF5 변환

**왜 변환이 필요한가?**
- Pickle: Python 전용, 느림, 보안 문제
- HDF5: 언어 독립적, 빠름, 대용량 데이터 효율적

**변환 스크립트**: `scripts/preprocessing/convert_camera_params.py`

```bash
python scripts/preprocessing/convert_camera_params.py \
  data/markerless_mouse_1_nerf/new_cam.pkl \
  data/markerless_mouse_1_nerf/camera_params.h5
```

**변환 결과**:
```python
# HDF5 구조
camera_params.h5
└── camera_parameters/
    ├── intrinsic: [6, 3, 3]    # 6개 카메라의 K
    ├── rotation: [6, 3, 3]     # 6개 카메라의 R
    └── translation: [6, 3]     # 6개 카메라의 T
```

---

## 전처리 단계별 설명

### **Step 0: 초기 설정**

#### 목적
카메라 파라미터를 output 디렉토리로 복사

#### 입력
```
data/markerless_mouse_1_nerf/camera_params.h5
```

#### 출력
```
output/markerless_mouse_nerf/camera_params.h5
```

#### 코드 설명
```bash
# 데이터 디렉토리에서 출력 디렉토리로 복사
cp data/markerless_mouse_1_nerf/camera_params.h5 \
   output/markerless_mouse_nerf/camera_params.h5
```

**왜 복사하는가?**
- 모든 중간 결과를 output 디렉토리에 모으기 위함
- 데이터 디렉토리는 읽기 전용 유지

---

### **Step 1: Up Direction 추정**

#### 목적
"위쪽" 방향을 자동으로 추정 (중력 반대 방향)

#### 입력
```
output/markerless_mouse_nerf/camera_params.h5
```

#### 출력
```
output/markerless_mouse_nerf/vertical_lines.npz
```

#### 원리
카메라들의 Y축 평균을 "위쪽"으로 설정

```python
# 각 카메라의 Y축 방향 추출
y_axes = rotation[:, :, 1]  # [6, 3] - 6개 카메라의 Y축

# 평균 계산 및 정규화
up = y_axes.mean(axis=0)    # [3] - 평균 Y축
up = up / np.linalg.norm(up)  # 단위 벡터로 정규화
```

**예시 출력**:
```
Estimated up direction: [0.023, 0.987, -0.159]
  - X: 0.023 (거의 0, 옆으로 기울지 않음)
  - Y: 0.987 (위쪽, 거의 1)
  - Z: -0.159 (약간 뒤쪽)
```

#### 왜 필요한가?
- 모든 카메라 뷰를 일관된 좌표계로 정렬
- 동물이 바닥에 서 있다고 가정 → "위쪽" 정의 필요

#### 수동 방식 (옵션)
GUI로 수직선을 그려서 추정 가능 (`estimate_up_direction.py`)

---

### **Step 2: Center & Rotation 계산**

#### 목적
각 프레임에서 동물의 **3D 중심점**과 **주축 방향** 계산

#### 입력
- `camera_params.h5`: 카메라 파라미터
- `vertical_lines.npz`: 위쪽 방향
- `simpleclick_undist/*.mp4`: 세그멘테이션 마스크 (6개 카메라)

#### 출력
```
output/markerless_mouse_nerf/center_rotation.npz
```

#### 원리: Shape-from-Silhouette (Visual Hull)

**1. 각 카메라의 2D 마스크를 3D 공간으로 역투영**

```python
# 2D 마스크 픽셀 (x, y) → 3D 공간의 광선(ray)
# 카메라에서 출발하여 무한대로 뻗는 선
```

**2. 6개 카메라의 광선들이 교차하는 영역 = 동물의 3D 볼륨**

```
Camera 1 마스크: ■■■■
Camera 2 마스크:   ■■■■
Camera 3 마스크:     ■■
...
→ 교집합 = 동물의 대략적 3D 형상
```

**3. 3D 볼륨의 중심(mean)과 주축(PCA) 계산**

```python
# 볼륨 내 모든 복셀(voxel)의 가중 평균
mean = np.sum(coordinates * volume_weights)

# 공분산 행렬의 고유벡터 → 주축 방향
eigenvalues, eigenvectors = np.linalg.eig(covariance)
# eigenvectors[:, 0] = 몸통의 긴 축 (머리-꼬리)
# eigenvectors[:, 1] = 두 번째 주축
# eigenvectors[:, 2] = 세 번째 주축
```

#### 출력 데이터 구조

```python
center_rotation.npz:
  - centers: [N, 3]      # N 프레임, 각 프레임의 3D 중심점 (x, y, z)
  - rotations: [N, 3, 3] # N 프레임, 각 프레임의 회전 행렬 (주축 방향)
```

**예시**:
```
Frame 0:
  Center: [0.05, 0.12, 0.03] meters  # 월드 좌표계
  Rotation: [[0.98, -0.15, 0.01],    # 몸통 긴 축
             [0.15,  0.98, 0.03],    # 옆 축
             [-0.02, 0.02, 0.99]]    # 위 축
```

#### 병렬 처리

18,000 프레임을 빠르게 처리하기 위해 **joblib** 사용:

```python
Parallel(n_jobs=16)(  # 16개 CPU 코어 사용
    delayed(process_chunk)(chunk) for chunk in chunks
)
```

---

### **Step 3: Crop Indices 계산**

#### 목적
3D 볼륨을 **관심 영역(ROI)**으로 잘라내는 범위 결정

#### 입력
- `camera_params.h5`
- `vertical_lines.npz`
- `center_rotation.npz`
- `simpleclick_undist/*.mp4`

#### 출력
```
output/markerless_mouse_nerf/volume_sum.npy
```

#### 원리

**1. 전체 프레임의 3D 볼륨 합산**

```python
# 모든 프레임에서 동물이 차지한 3D 공간을 누적
total_volume = sum(volume[frame] for frame in all_frames)
```

**2. 빈 공간 제거**

```python
# X, Y, Z 축 각각에서 동물이 있는 범위만 선택
x_range = [x_min, x_max]  # 예: [10, 90] (인덱스)
y_range = [y_min, y_max]  # 예: [20, 95]
z_range = [z_min, z_max]  # 예: [25, 88]
```

**3. Config 파일에 저장**

```json
{
  "volume_idx": [[10, 90], [20, 95], [25, 88]]
}
```

#### 왜 필요한가?

**메모리 절약**:
```
전체 그리드: 112 × 112 × 112 = 1.4M 복셀
크롭 후: 80 × 75 × 63 = 378K 복셀 (73% 감소!)
```

**학습 효율**:
- 빈 공간을 학습하지 않음
- GPU 메모리 절약
- 학습 속도 향상

---

### **Step 4: Write Images (HDF5)**

#### 목적
**비디오 프레임**을 빠르게 읽을 수 있는 **HDF5 파일**로 변환

#### 입력
- `videos_undist/*.mp4`: 6개 카메라 RGB 비디오
- `simpleclick_undist/*.mp4`: 6개 카메라 마스크
- `center_rotation.npz`: 각 프레임의 3D 정보
- Config의 `frame_jump`: 5 (매 5프레임마다 샘플링)

#### 출력
```
output/markerless_mouse_nerf/images/images.h5
```

#### 처리 과정

**1. 프레임 샘플링**

```python
total_frames = 18,000
frame_jump = 5
selected_frames = [0, 5, 10, 15, ..., 17,995]  # 3,600 프레임
```

**2. 이미지 다운샘플링**

```python
original_size = (1152, 1024)
downsample = 4
new_size = (1152//4, 1024//4) = (288, 256)
```

**3. 정규화 및 크롭**

```python
# 각 프레임을 동물 중심으로 정렬 및 회전
rotated_image = rotate_by_rotation_matrix(image, R[frame])
cropped_image = crop_by_center(rotated_image, center[frame])
normalized_image = (image / 255.0).astype(float32)
```

**4. HDF5 저장**

```python
images.h5:
  - images: [3600, 6, 256, 288, 3]  # NCHW 형식
    ├─ N=3600: 프레임 수
    ├─ C=6: 카메라 수
    ├─ H=256: 높이
    ├─ W=288: 너비
    └─ 3: RGB 채널

  - masks: [3600, 6, 256, 288]      # 세그멘테이션 마스크
```

#### 병렬 처리

```python
# 3600 프레임을 16개 청크로 분할
chunk_size = 3600 / 16 = 225 frames/chunk

Parallel(n_jobs=16)(
    delayed(process_chunk)(chunk) for chunk in chunks
)
```

#### 왜 비디오 대신 HDF5?

| 특징 | MP4 비디오 | HDF5 파일 |
|------|-----------|----------|
| **랜덤 액세스** | 느림 (순차 디코딩) | 빠름 (직접 접근) |
| **압축** | 손실 압축 | 무손실 압축 |
| **학습 속도** | 느림 | 빠름 (10배+) |
| **메모리** | 많이 사용 | 효율적 |

---

### **Step 5: Copy to ZARR**

#### 목적
HDF5를 **ZARR 형식**으로 변환 (PyTorch 학습 최적화)

#### 입력
```
output/markerless_mouse_nerf/images/images.h5
```

#### 출력
```
output/markerless_mouse_nerf/images/images.zarr/
```

#### ZARR vs HDF5

| 특징 | HDF5 | ZARR |
|------|------|------|
| **병렬 읽기** | 제한적 | 우수 |
| **클라우드 스토리지** | 어려움 | 쉬움 |
| **청크 단위** | 고정 | 유연 |
| **PyTorch 통합** | 보통 | 우수 |

#### ZARR 디렉토리 구조

```
images.zarr/
├── images/
│   ├── 0.0.0        # 청크 파일
│   ├── 0.0.1
│   └── ...
├── masks/
│   ├── 0.0.0
│   └── ...
└── .zarray          # 메타데이터
```

#### 변환 코드

```python
import zarr
import h5py

# HDF5 읽기
with h5py.File('images.h5', 'r') as h5f:
    images = h5f['images'][:]
    masks = h5f['masks'][:]

# ZARR 쓰기 (청크 단위 압축)
zarr.save('images.zarr', images=images, masks=masks,
          chunks=(10, 1, 256, 288, 3))  # 10프레임씩 청크
```

---

## 실전 예제

### 전체 파이프라인 실행

```bash
# 1. 디렉토리 이동
cd /home/joon/pose-splatter

# 2. Conda 환경 활성화
conda activate splatter

# 3. 의존성 설치
pip install opencv-python h5py zarr joblib tqdm

# 4. 환경 변수 설정
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5. 전처리 실행 (자동화 스크립트)
bash scripts/preprocessing/run_full_preprocessing.sh \
  configs/baseline/markerless_mouse_nerf.json
```

### 수동 실행 (각 단계별)

```bash
PROJECT="output/markerless_mouse_nerf"
DATA="data/markerless_mouse_1_nerf"
CONFIG="configs/baseline/markerless_mouse_nerf.json"

# Step 0: 카메라 파라미터 복사
cp $DATA/camera_params.h5 $PROJECT/camera_params.h5

# Step 1: Up direction 추정
python scripts/preprocessing/auto_estimate_up.py \
  $PROJECT/camera_params.h5 \
  $PROJECT/vertical_lines.npz

# Step 2: Center & Rotation
python scripts/preprocessing/calculate_center_rotation.py $CONFIG

# Step 3: Crop indices
python scripts/preprocessing/calculate_crop_indices.py $CONFIG

# Step 4: Write images
python scripts/preprocessing/write_images.py $CONFIG

# Step 5: Copy to ZARR
python scripts/preprocessing/copy_to_zarr.py \
  $PROJECT/images/images.h5 \
  $PROJECT/images/images.zarr
```

---

## 문제 해결

### 문제 1: `ModuleNotFoundError: No module named 'cv2'`

**원인**: opencv-python 미설치

**해결**:
```bash
conda activate splatter
pip install opencv-python
```

### 문제 2: `FileNotFoundError: camera_params.h5`

**원인**: 카메라 파라미터가 없음

**해결**:
```bash
# new_cam.pkl → camera_params.h5 변환
python scripts/preprocessing/convert_camera_params.py \
  data/markerless_mouse_1_nerf/new_cam.pkl \
  data/markerless_mouse_1_nerf/camera_params.h5
```

### 문제 3: `total_frames: 0`

**원인**: 이전 단계 실패로 center_rotation.npz가 없음

**해결**:
```bash
# 출력 디렉토리 삭제 후 재실행
rm -rf output/markerless_mouse_nerf/
bash scripts/preprocessing/run_full_preprocessing.sh $CONFIG
```

### 문제 4: CUDA OOM (메모리 부족)

**원인**: GPU 메모리 부족

**해결**:
```bash
# Config 파일에서 image_downsample 증가
{
  "image_downsample": 6,  # 4 → 6 (이미지 크기 감소)
  "grid_size": 96,        # 112 → 96 (볼륨 해상도 감소)
}
```

---

## 요약

| 단계 | 입력 | 출력 | 소요 시간 | 목적 |
|------|------|------|-----------|------|
| **0** | `camera_params.h5` (data) | `camera_params.h5` (output) | 1초 | 파일 복사 |
| **1** | `camera_params.h5` | `vertical_lines.npz` | 1초 | 위쪽 방향 |
| **2** | 마스크 비디오 | `center_rotation.npz` | 5-10분 | 3D 중심/회전 |
| **3** | 볼륨 데이터 | `volume_sum.npy` | 3-5분 | 크롭 범위 |
| **4** | RGB/마스크 비디오 | `images.h5` | 10-20분 | HDF5 변환 |
| **5** | `images.h5` | `images.zarr` | 2-5분 | ZARR 변환 |
| **전체** | - | - | **20-40분** | - |

---

## 다음 단계

전처리 완료 후:

```bash
# 1. 데이터 검증
python scripts/utils/verify_datasets.py

# 2. 학습 시작
python scripts/training/train_script.py $CONFIG --epochs 50

# 3. 학습 모니터링
tail -f output/markerless_mouse_nerf/logs/step6_training.log
```
