# 카메라 좌표계와 위쪽 방향 추정 - 완전 가이드

## 목차
1. [new_cam.pkl 파일 구조](#newcampkl-파일-구조)
2. [HDF5 변환 이유와 과정](#hdf5-변환-이유와-과정)
3. [좌표계의 종류](#좌표계의-종류)
4. [위쪽 방향 추정 이유](#위쪽-방향-추정-이유)
5. [좌표계 정렬 과정](#좌표계-정렬-과정)
6. [실전 예제](#실전-예제)

---

## new_cam.pkl 파일 구조

### 1. 파일 읽기

```python
import pickle
import numpy as np

# Pickle 파일 로드
with open('new_cam.pkl', 'rb') as f:
    cameras = pickle.load(f)

print(type(cameras))  # <class 'list'>
print(len(cameras))   # 6 (6개 카메라)
```

### 2. 데이터 구조

```python
cameras = [
    {  # Camera 0
        'K': np.array([[fx, 0, cx],      # Intrinsic matrix [3×3]
                       [0, fy, cy],
                       [0,  0,  1]]),

        'R': np.array([[r11, r12, r13],  # Rotation matrix [3×3]
                       [r21, r22, r23],
                       [r31, r32, r33]]),

        'T': np.array([tx, ty, tz]),     # Translation vector [3]

        'mapx': np.array([...]),         # X distortion map [1024×1152]
        'mapy': np.array([...])          # Y distortion map [1024×1152]
    },
    # ... Camera 1~5
]
```

### 3. 각 요소 상세 분석

#### **K (Intrinsic Matrix)** - 카메라 내부 파라미터

```python
K = [[1632.31,    0.00,  601.35],
     [   0.00, 1639.30,  491.22],
     [   0.00,    0.00,    1.00]]

# 실제 값 (Camera 0)
fx = 1632.31  # X축 초점 거리 (픽셀)
fy = 1639.30  # Y축 초점 거리 (픽셀)
cx = 601.35   # 주점 X 좌표 (픽셀)
cy = 491.22   # 주점 Y 좌표 (픽셀)
```

**물리적 의미**:

1. **초점 거리 (f_x, f_y)**
   ```
   초점 거리 (mm) = fx × 픽셀 크기 (mm/pixel)

   예: 센서 크기 = 1/2.3" (6.17 × 4.55 mm)
       픽셀 수 = 1152 × 1024
       픽셀 크기 = 6.17 / 1152 ≈ 0.00535 mm/pixel

   → 실제 초점 거리 = 1632 × 0.00535 ≈ 8.7 mm
   ```

2. **주점 (c_x, c_y)**
   - 렌즈 광학 중심이 센서와 만나는 점
   - 이상적으로는 이미지 중심: (1152/2, 1024/2) = (576, 512)
   - 실제: (601, 491) → 약간 오른쪽 위로 치우침 (제조 오차)

**3D → 2D 투영 공식**:

```python
# 월드 좌표 (X, Y, Z) → 이미지 픽셀 (u, v)

# Step 1: 카메라 좌표로 변환
P_cam = R @ P_world + T

# Step 2: 정규화 평면에 투영
x = P_cam[0] / P_cam[2]
y = P_cam[1] / P_cam[2]

# Step 3: 픽셀 좌표로 변환
u = fx * x + cx
v = fy * y + cy
```

---

#### **R (Rotation Matrix)** - 카메라 방향

```python
R = [[+0.577, -0.817, -0.014],
     [+0.003, +0.019, -1.000],
     [+0.817, +0.577, +0.013]]

# 각 열은 카메라 축의 방향
X_axis = R[:, 0] = [+0.577, +0.003, +0.817]  # 카메라 오른쪽
Y_axis = R[:, 1] = [-0.817, +0.019, +0.577]  # 카메라 아래쪽
Z_axis = R[:, 2] = [-0.014, -1.000, +0.013]  # 카메라 앞쪽 (viewing direction)
```

**회전 행렬의 성질**:

1. **정규 직교 행렬 (Orthonormal)**
   ```python
   # 각 열은 단위 벡터
   np.linalg.norm(R[:, 0]) ≈ 1.0
   np.linalg.norm(R[:, 1]) ≈ 1.0
   np.linalg.norm(R[:, 2]) ≈ 1.0

   # 열끼리 직교
   R[:, 0] @ R[:, 1] ≈ 0
   R[:, 0] @ R[:, 2] ≈ 0
   R[:, 1] @ R[:, 2] ≈ 0

   # R @ R^T = I
   R @ R.T ≈ [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]
   ```

2. **행렬식 = +1**
   ```python
   det(R) = +1.0  # Proper rotation (no reflection)
   ```

**카메라 좌표계**:

```
      Y (down)
      |
      |
      +---- X (right)
     /
    /
   Z (forward, into the scene)
```

**월드 좌표계 → 카메라 좌표계 변환**:

```python
P_camera = R @ P_world + T

# 역변환 (카메라 → 월드)
P_world = R.T @ (P_camera - T)
        = R.T @ P_camera - R.T @ T
```

---

#### **T (Translation Vector)** - 카메라 위치

```python
T = [+10.34, +66.41, +236.70]  # Camera 0

# 단위: 센티미터 (cm)
# DANNCE 데이터셋의 관례
```

**주의**: `T`는 카메라 위치가 **아닙니다**!

```python
# T는 월드 원점을 카메라 좌표계로 변환할 때의 벡터
# 카메라 중심 (Camera Center)은 다름:

C = -R.T @ T

# Camera 0 예시:
C = -R.T @ [10.34, 66.41, 236.70]
  = [-199.48, -129.34, +63.44]  # 실제 카메라 위치 (cm)
```

**투영 행렬 (Projection Matrix)**:

```python
# 월드 3D → 이미지 2D 한 번에
P = K @ [R | T]
  = K @ [R @ I | T]
  = [[fx, 0, cx],     [[r11, r12, r13, tx],
     [0, fy, cy],  @   [r21, r22, r23, ty],
     [0,  0,  1]]      [r31, r32, r33, tz]]

# 사용:
p_image = P @ [X, Y, Z, 1]^T
p_image = p_image / p_image[2]  # 정규화
u, v = p_image[0], p_image[1]
```

---

#### **mapx, mapy (Distortion Maps)** - 렌즈 왜곡 보정

```python
mapx.shape = (1024, 1152)  # 이미지와 동일 크기
mapy.shape = (1024, 1152)

# 왜곡된 이미지 → 보정된 이미지
import cv2
undistorted = cv2.remap(distorted_image, mapx, mapy, cv2.INTER_LINEAR)
```

**왜곡의 종류**:

1. **방사 왜곡 (Radial Distortion)**
   - 렌즈 중심에서 멀수록 왜곡 증가
   - Barrel distortion: 광각 렌즈 (물고기 눈)
   - Pincushion distortion: 망원 렌즈

2. **접선 왜곡 (Tangential Distortion)**
   - 렌즈와 센서가 평행하지 않을 때

**왜 미리 계산?**

```python
# 매 프레임마다 계산 (느림)
for frame in video:
    undistorted = undistort_with_coefficients(frame, k1, k2, p1, p2, k3)

# 한 번만 계산 (빠름)
mapx, mapy = cv2.initUndistortRectifyMap(K, dist_coeffs, ...)
for frame in video:
    undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
```

---

## HDF5 변환 이유와 과정

### 1. 왜 HDF5로 변환하는가?

#### **파일 크기 비교**

```
new_cam.pkl:      54.00 MB
camera_params.h5:  0.005 MB  (5 KB!)

감소율: 99.99%
```

**크기 차이의 이유**:

```python
# PKL 파일 내용:
- K: 6 × 3 × 3 × 8 bytes = 432 bytes
- R: 6 × 3 × 3 × 8 bytes = 432 bytes
- T: 6 × 3 × 8 bytes = 144 bytes
- mapx: 6 × 1024 × 1152 × 4 bytes = 28.3 MB
- mapy: 6 × 1024 × 1152 × 4 bytes = 28.3 MB
────────────────────────────────────────────
Total: ~56.6 MB

# HDF5 파일 내용 (mapx, mapy 제외):
- intrinsic: 432 bytes
- rotation: 432 bytes
- translation: 144 bytes
────────────────────────────────
Total: ~1 KB + 메타데이터
```

**왜 mapx, mapy를 제외?**

- 이미 `videos_undist/` 비디오가 보정됨
- Distortion maps는 원본 비디오 보정용
- 우리는 이미 보정된 비디오 사용 → 불필요

#### **보안 문제**

```python
# Pickle: 임의의 Python 코드 실행 가능
import pickle
import os

class Malicious:
    def __reduce__(self):
        return (os.system, ('rm -rf /',))

# 이 pickle 로드 시 컴퓨터 전체 삭제!
pickle.dump(Malicious(), open('evil.pkl', 'wb'))

# HDF5: 순수 데이터만, 코드 실행 불가능
```

#### **성능 비교**

| 작업 | Pickle | HDF5 |
|------|--------|------|
| **전체 로드** | 2.3 초 | 0.02 초 |
| **부분 로드** | 불가능 | 0.001 초 |
| **메모리** | 56 MB | 1 KB |
| **다른 언어** | Python만 | C, MATLAB, R, Julia... |

### 2. 변환 과정

```python
import pickle
import h5py
import numpy as np

# Step 1: Pickle 읽기
with open('new_cam.pkl', 'rb') as f:
    cameras = pickle.load(f)

# Step 2: 데이터 추출
intrinsics = []
rotations = []
translations = []

for cam in cameras:
    intrinsics.append(cam['K'])
    rotations.append(cam['R'])

    # T를 1D로 변환
    T = cam['T']
    if T.ndim == 2:
        T = T.flatten()
    translations.append(T)

# Step 3: NumPy 배열로 변환
intrinsics = np.array(intrinsics)    # [6, 3, 3]
rotations = np.array(rotations)      # [6, 3, 3]
translations = np.array(translations) # [6, 3]

# Step 4: HDF5 저장
with h5py.File('camera_params.h5', 'w') as f:
    grp = f.create_group('camera_parameters')
    grp.create_dataset('intrinsic', data=intrinsics, compression='gzip')
    grp.create_dataset('rotation', data=rotations, compression='gzip')
    grp.create_dataset('translation', data=translations, compression='gzip')
```

### 3. HDF5 파일 읽기

```python
import h5py

with h5py.File('camera_params.h5', 'r') as f:
    # 전체 로드
    K_all = f['camera_parameters']['intrinsic'][:]  # [6, 3, 3]

    # 부분 로드 (Camera 0만)
    K_0 = f['camera_parameters']['intrinsic'][0]    # [3, 3]

    # Lazy loading (메모리 절약)
    K_lazy = f['camera_parameters']['intrinsic']    # 아직 메모리에 안 올림
    K_0 = K_lazy[0]  # 이때 로드
```

---

## 좌표계의 종류

### 1. 월드 좌표계 (World Coordinate System)

**정의**: 실험실 공간의 절대 좌표계

```
       Z (up)
       |
       |
       +---- X (right)
      /
     /
    Y (forward)
```

**설정 방법**:
- DANNCE: 캘리브레이션 보드의 코너 중 하나를 원점으로 설정
- 원점: 보통 바닥의 특정 지점

**단위**: 센티미터 (cm)

**예시**:
```python
mouse_position = [5.2, 12.3, 2.1]  # cm
# X = 5.2 cm 오른쪽
# Y = 12.3 cm 앞쪽
# Z = 2.1 cm 위쪽
```

---

### 2. 카메라 좌표계 (Camera Coordinate System)

**정의**: 각 카메라를 원점으로 하는 좌표계 (6개 카메라 = 6개 좌표계)

```
      Y (down)
      |
      |
      +---- X (right)
     /
    /
   Z (forward, viewing direction)
```

**특징**:
- 원점: 카메라 렌즈 중심
- Z축: 카메라가 바라보는 방향 (viewing direction)
- Y축: 이미지에서 아래쪽

**변환 공식**:
```python
# 월드 → 카메라
P_cam = R @ P_world + T

# 카메라 → 월드
P_world = R.T @ (P_cam - T)
```

---

### 3. 이미지 좌표계 (Image/Pixel Coordinate System)

**정의**: 2D 이미지 평면의 픽셀 좌표

```
(0, 0) ───────> u (width, 1152)
  |
  |
  |
  v
  v (height, 1024)
```

**단위**: 픽셀

**변환 공식**:
```python
# 카메라 3D → 이미지 2D
[x, y, z]_cam → [u, v]_image

u = (fx * x / z) + cx
v = (fy * y / z) + cy
```

---

### 4. 정규화 좌표계 (Normalized Coordinate System)

**정의**: 카메라 내부 파라미터를 제거한 좌표계

```python
# 카메라 좌표 정규화
x_norm = X_cam / Z_cam
y_norm = Y_cam / Z_cam

# 단위: 무차원 (dimensionless)
```

**용도**:
- 다른 카메라와 비교
- Epipolar geometry 계산

---

## 위쪽 방향 추정 이유

### 1. 문제: 회전 불확정성 (Rotation Ambiguity)

**카메라 캘리브레이션의 한계**:

```python
# 6개 카메라가 동물을 둘러싸고 있음
# 하지만, 전체 시스템이 Z축으로 회전해도 카메라 간 상대 위치는 동일!

# Configuration 1 (original)
cameras = [C0, C1, C2, C3, C4, C5]

# Configuration 2 (rotated 45° around Z)
cameras_rotated = [rotate_z(C0, 45°), rotate_z(C1, 45°), ...]

# 카메라끼리는 구분 불가!
```

**결과**:
- 월드 좌표계의 "위쪽"이 정의되지 않음
- 재구성된 3D 동물이 기울어져 있을 수 있음
- 학습 불안정 (매 epoch마다 다른 방향)

### 2. 해결책: Up Direction 명시

```python
# 물리적 제약 활용
# → 바닥은 아래, 중력은 아래, 동물은 바닥 위

up_direction = [0, 0, 1]  # Z축이 위쪽
```

**효과**:
1. **좌표계 일관성**: 모든 프레임에서 동일한 "위쪽"
2. **학습 안정성**: 네트워크가 일관된 좌표 학습
3. **시각화 편의**: 항상 Z축이 위쪽

### 3. 자동 추정 방법

```python
def auto_estimate_up(cameras):
    """
    카메라 Y축의 평균 = 아래쪽
    → 그 반대 = 위쪽
    """
    y_axes = []
    for cam in cameras:
        R = cam['R']
        y_axis = R[:, 1]  # 카메라 Y축 (이미지 아래 방향)
        y_axes.append(y_axis)

    # 평균 계산
    down = np.array(y_axes).mean(axis=0)

    # 반대 방향 = 위쪽
    up = -down
    up = up / np.linalg.norm(up)  # 정규화

    return up

# 예시 출력:
# up = [0.023, 0.987, -0.159]
#      ≈ [0, 1, 0]  (거의 Y축)
```

**왜 Y축의 평균?**

1. 카메라는 보통 바닥과 평행하게 설치
2. 카메라 Y축 = 이미지 아래 = 중력 방향
3. 6개 카메라 평균 → 노이즈 제거

---

## 좌표계 정렬 과정

### 1. 원래 상태 (정렬 전)

```
카메라들의 월드 좌표계:
- 임의로 회전되어 있음
- "위쪽"이 일정하지 않음
- 각 프레임마다 동물 방향이 다름

Frame 0: 동물이 X축 방향
Frame 1: 동물이 Y축 방향  (학습 어려움!)
Frame 2: 동물이 Z축 방향
```

### 2. Step 1: Up Direction 추정

```python
# 출력: vertical_lines.npz
up = [0.023, 0.987, -0.159]

# 이제 "위쪽"이 정의됨!
```

### 3. Step 2: 각 프레임을 정렬된 좌표계로 회전

```python
# 각 프레임에서:
# 1. 동물 중심을 원점으로 이동
# 2. 동물의 "위쪽"을 월드 Z축으로 회전
# 3. 동물의 "앞쪽"을 월드 Y축으로 회전

# center_rotation.npz 저장:
centers[frame] = [cx, cy, cz]  # 동물 중심
rotations[frame] = R_align      # 정렬 회전 행렬
```

**정렬 회전 행렬 계산**:

```python
def compute_alignment_rotation(volume, up_direction):
    """
    동물의 3D 볼륨 → 정렬 회전 행렬
    """
    # Step 1: 볼륨의 주축 (PCA)
    covariance = compute_covariance(volume)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # 고유값 큰 순서로 정렬
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # 주축 방향
    long_axis = eigenvectors[:, 0]   # 머리-꼬리
    medium_axis = eigenvectors[:, 1]
    short_axis = eigenvectors[:, 2]

    # Step 2: Up direction과 정렬
    # short_axis를 up_direction으로 회전
    R_align = align_vectors(short_axis, up_direction)

    return R_align
```

### 4. 정렬 후

```
모든 프레임:
- 동물 중심 = 원점
- 동물 위쪽 = Z축
- 동물 앞쪽 = Y축
- 동물 옆 = X축

→ 일관된 좌표계!
→ 네트워크 학습 쉬워짐
```

**시각화**:

```
정렬 전:
Frame 0: ↗ (동물이 대각선)
Frame 1: → (동물이 오른쪽)
Frame 2: ↑ (동물이 위쪽)

정렬 후:
Frame 0: ↑ (항상 위쪽)
Frame 1: ↑
Frame 2: ↑
```

---

## 실전 예제

### 1. Camera Parameters 분석

```bash
# 분석 스크립트 실행
python scripts/utils/analyze_camera_params.py \
  data/markerless_mouse_1_nerf/new_cam.pkl \
  data/markerless_mouse_1_nerf/camera_params.h5

# 출력:
# - 각 카메라의 K, R, T 값
# - 카메라 위치 시각화
# - PKL vs HDF5 비교
# - new_cam_visualization.png 생성
```

### 2. 수동 변환

```python
import pickle
import h5py
import numpy as np

# PKL 읽기
with open('data/markerless_mouse_1_nerf/new_cam.pkl', 'rb') as f:
    cameras = pickle.load(f)

# 카메라 0 분석
cam0 = cameras[0]
print("Keys:", cam0.keys())
print("K shape:", cam0['K'].shape)
print("R shape:", cam0['R'].shape)
print("T shape:", cam0['T'].shape)
print("mapx shape:", cam0['mapx'].shape)
print("mapy shape:", cam0['mapy'].shape)

# 카메라 중심 계산
R = cam0['R']
T = cam0['T']
C = -R.T @ T
print(f"Camera 0 center: {C}")

# HDF5로 변환
python scripts/preprocessing/convert_camera_params.py \
  data/markerless_mouse_1_nerf/new_cam.pkl \
  data/markerless_mouse_1_nerf/camera_params.h5
```

### 3. Up Direction 추정

```python
import h5py
import numpy as np

# HDF5 읽기
with h5py.File('output/markerless_mouse_nerf/camera_params.h5', 'r') as f:
    rotations = f['camera_parameters']['rotation'][:]  # [6, 3, 3]

# Y축 추출 (카메라 아래 방향)
y_axes = rotations[:, :, 1]  # [6, 3]

# 평균 계산
down = y_axes.mean(axis=0)

# 반대 = 위쪽
up = -down
up = up / np.linalg.norm(up)

print(f"Estimated up direction: {up}")
# [0.023, 0.987, -0.159] ≈ Y축
```

### 4. 좌표 변환 예제

```python
# 월드 좌표의 점
P_world = np.array([10.0, 20.0, 5.0])  # cm

# Camera 0로 투영
cam0 = cameras[0]
K = cam0['K']
R = cam0['R']
T = cam0['T']

# 1. 카메라 좌표로 변환
P_cam = R @ P_world + T
print(f"Camera coords: {P_cam}")

# 2. 이미지 평면에 투영
x = P_cam[0] / P_cam[2]
y = P_cam[1] / P_cam[2]

# 3. 픽셀 좌표
u = K[0, 0] * x + K[0, 2]
v = K[1, 1] * y + K[1, 2]
print(f"Pixel coords: ({u:.1f}, {v:.1f})")
```

---

## 요약

| 질문 | 답변 |
|------|------|
| **new_cam.pkl 구조?** | List of dicts, 각각 K, R, T, mapx, mapy |
| **어떻게 읽나?** | `pickle.load()` |
| **각 요소 이름?** | K=내부, R=회전, T=이동, mapx/mapy=왜곡보정 |
| **HDF5 변환 이유?** | 99.99% 작음, 빠름, 안전, 언어 독립적 |
| **변환 과정?** | PKL→NumPy→HDF5, mapx/mapy 제외 |
| **위쪽 추정 이유?** | 회전 불확정성 제거, 학습 안정화 |
| **좌표계 정렬?** | 모든 프레임을 일관된 방향으로 회전 |

**핵심**:
1. 카메라 파라미터 = 3D ↔ 2D 변환 정보
2. HDF5 = 효율적 저장
3. Up direction = 좌표계 일관성
4. 정렬 = 학습 안정화
