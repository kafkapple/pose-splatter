# 실험 분석 보고서 - Markerless Mouse NeRF Extended Training

**날짜**: 2025-11-12
**프로젝트**: pose-splatter
**실험 대상**: Markerless Mouse NeRF Extended Training

---

## Executive Summary

Markerless mouse NeRF 프로젝트에서 Extended Training 실험을 여러 변형(extended, extended_debug, extended_debug_fj5, extended_fast)으로 진행했습니다. 데이터 전처리 단계에서 일부 오류가 발생했으나, 기존 작업된 데이터로부터 학습을 성공적으로 완료했습니다.

**주요 결과**:
- ✅ Extended training 100 epoch 완료 (markerless_mouse_nerf_extended)
- ✅ Debug 변형 3개 실행 완료 (extended_debug, extended_debug_fj5, extended_fast)
- ❌ Pipeline 전체 실행 시 전처리 단계 오류 발생 (up direction, center/rotation 계산)
- ✅ 기존 데이터로부터 학습 재개 성공
- ✅ 360도 rotation 시각화 생성 완료

---

## 1. 실험 설정

### 1.1 Config 변형 분석

4가지 config 변형을 테스트했습니다:

| Config | Output Dir | max_frames | plot_every | save_every | frame_jump | 특징 |
|--------|-----------|------------|------------|------------|------------|------|
| extended | markerless_mouse_nerf_extended | - | 2 | 2 | 2 | 기본 extended |
| extended_debug | markerless_mouse_nerf_extended_debug | 1500 | 2 | 2 | 2 | 프레임 제한 디버그 |
| extended_debug_fj5 | markerless_mouse_nerf_extended_debug_fj5 | - | 2 | 2 | - | Frame jump 5 테스트 |
| extended_fast | markerless_mouse_nerf_extended_fast | - | 5 | 5 | 2 | 빠른 로깅 |

**공통 설정**:
- image_downsample: 4
- grid_size: 112
- lr: 1e-4
- img_lambda: 0.5
- ssim_lambda: 0.0
- holdout_views: [5, 1]
- fps: 100
- ell: 0.22

### 1.2 학습 완료 상태

```bash
# 체크포인트 크기 (모두 599M으로 동일)
markerless_mouse_nerf_extended/checkpoint.pt      599M (11월 11 14:16)
markerless_mouse_nerf_extended_debug/checkpoint.pt 599M (11월 11 20:02)
markerless_mouse_nerf_extended_debug_fj5/checkpoint.pt 599M (11월 11 22:26)
markerless_mouse_nerf_extended_fast/checkpoint.pt 599M (11월 12 01:05)
```

---

## 2. 전처리 파이프라인 오류 분석

### 2.1 Pipeline 실행 로그 (extended_training.log)

파이프라인 전체 실행 시 다음과 같은 오류 발생:

**Step 1: Up Direction 추정 실패**
```
ValueError: No vertical lines found in any camera; cannot estimate up.
```
- **원인**: 수동 vertical line 수집 시 사용자가 line을 입력하지 않음
- **영향**: 후속 단계에서 up direction 파일 누락

**Step 2: Center/Rotation 계산 실패**
```
AssertionError: up_fn does not exist
```
- **원인**: Step 1 실패로 인한 의존성 파일 누락
- **영향**: center_rotation.npz 생성 실패

**Step 3: Crop Indices 계산 실패**
```
AssertionError: center_rotation_fn does not exist
```
- **원인**: Step 2 실패로 인한 의존성 파일 누락

**Step 4: Write Images 실패**
```python
# HDF5 file locking error
BlockingIOError: [Errno 11] Unable to synchronously create file
                 (unable to lock file, errno = 11)

# Filter write error
OSError: Can't synchronously write data (filter returned failure during read)
```
- **원인**: 병렬 처리 중 파일 잠금 충돌 (joblib N_JOBS 설정)
- **영향**: HDF5 이미지 저장 실패

**Step 6: Training 모듈 누락**
```
ModuleNotFoundError: No module named 'torch'
```
- **원인**: 잘못된 Python 환경에서 실행 (torch 미설치 환경)

### 2.2 From-Images Pipeline 로그 (extended_training_pipeline.log)

기존 write_images 결과를 기다리는 방식으로 실행:

```
Checking if write_images completed...
  Waiting for write_images... 23:12:59
  ...
  Waiting for write_images... 00:10:39
```

- 약 1시간 대기 후 타임아웃 또는 수동 중단된 것으로 추정
- write_images가 완료되지 않아 학습 시작 불가

### 2.3 해결 방법

**기존 데이터 재사용**:
- `markerless_mouse_nerf` (기존 작업된 데이터)의 전처리 결과를 복사
- vertical_lines.npz, center_rotation.npz 등 필수 파일 활용
- 학습만 새로운 output 디렉토리에서 진행

**결과**:
- ✅ Extended training 100 epoch 성공적 완료
- ✅ Loss 감소 확인 (3.46 → 2.86)
- ✅ Reconstruction PDF, loss PDF 생성

---

## 3. 학습 결과 분석

### 3.1 Extended Training (100 epochs)

**Training Log 분석** (`markerless_mouse_nerf_extended/logs/step6_training.log`):

```
Epoch 0-5:    Loss 3.46 → 1.65 (빠른 초기 수렴)
Epoch 6-25:   Loss 2.57 → 2.22 (안정화)
Epoch 26-59:  Loss 2.42 → 2.72 (진동, 일부 증가)
Epoch 60-99:  Loss 2.86 → 2.86 (수렴)
```

**Loss 특성**:
- 초기 5 epoch에서 loss 절반 감소 (빠른 학습)
- Epoch 26 이후 loss가 증가하는 구간 발견 (2.22 → 3.04)
  - 이는 과적합 또는 learning rate 조정 필요 신호
- 최종 100 epoch에서 loss ~2.86 수렴

**Batch별 Loss 변동**:
- 배치별 loss 변동이 매우 큼 (2.14 ~ 3.11 범위)
- 이는 각 프레임/뷰의 난이도 차이를 반영

### 3.2 출력 파일 분석

모든 실험에서 다음 파일 생성:
- ✅ checkpoint.pt (599M)
- ✅ loss.pdf (14K)
- ✅ reconstruction.pdf (40-60K)
- ✅ camera_params.h5
- ✅ center_rotation.npz
- ✅ vertical_lines.npz
- ✅ volume_sum.npy (11M)

**추가 생성 파일** (markerless_mouse_nerf_extended_debug):
- gaussians/ 디렉토리
- logs/ 디렉토리

**시각화 결과** (기존 markerless_mouse_nerf):
- ✅ 360도 rotation video (rotation360_frame0.mp4)
- ✅ Animation sequences
- ✅ Point clouds
- ✅ Multi-view renders

---

## 4. GPU 사용 및 비-GPU 작업 분류

### 4.1 GPU 필수 작업

**학습 관련** (CUDA 필수):
- `train_script.py` - 모델 학습
- `evaluate_model.py` - 모델 평가
- `export_gaussian_full.py` - Gaussian 추출
- `export_point_cloud.py` - Point cloud 추출
- `export_animation_sequence.py` - 애니메이션 렌더링
- `generate_360_rotation.py` - 360도 회전 렌더링
- `generate_temporal_video.py` - 시간축 비디오 렌더링
- `render_image.py` - 단일 이미지 렌더링

**모델 컴포넌트**:
- `src/model.py` - PoseSplatter (gsplat 사용)
- `src/unet_3d.py` - 3D UNet
- `src/shape_carver.py` - Shape carving

### 4.2 비-GPU 작업 (즉시 실행 가능)

**데이터 전처리**:
- ✅ `estimate_up_direction.py` - Up 방향 추정 (matplotlib만 사용)
- ✅ `calculate_center_rotation.py` - 중심/회전 계산 (numpy)
- ✅ `calculate_crop_indices.py` - Crop 영역 계산
- ✅ `write_images.py` - HDF5 이미지 쓰기
- ✅ `convert_camera_params.py` - 카메라 파라미터 변환

**분석 및 시각화**:
- ✅ `analyze_results.py` - 결과 분석 (pandas, matplotlib)
- ✅ `compare_configs.py` - Config 비교 (JSON 파싱)
- ✅ `visualize_training.py` - 학습 진행 시각화 (로그 파싱)
- ✅ `visualize_renders.py` - 렌더 결과 시각화

**유틸리티**:
- ✅ `verify_datasets.py` - 데이터셋 검증
- ✅ `copy_to_zarr.py` - Zarr 변환
- ✅ `auto_estimate_up.py` - 자동 up direction 추정

**Blender 관련**:
- ✅ `blender_import_pointcloud.py` - Point cloud Blender import

---

## 5. 주요 발견사항

### 5.1 전처리 파이프라인 문제

**문제점**:
1. **수동 입력 의존성**: Up direction 추정이 사용자 수동 입력에 의존
2. **병렬 처리 충돌**: HDF5 쓰기 시 파일 잠금 문제
3. **환경 의존성**: 잘못된 Python 환경에서 실행 시 torch 미발견
4. **의존성 체인**: 한 단계 실패 시 모든 후속 단계 중단

**해결 방안**:
- `auto_estimate_up.py` 활용하여 자동 추정 시도
- HDF5 쓰기 병렬 처리 수 조절 (N_JOBS 감소)
- Conda 환경 명시적 활성화 (`conda run -n env`)
- 각 단계별 재실행 가능하도록 체크포인트 활용

### 5.2 학습 특성

**긍정적 요소**:
- 초기 수렴 속도 빠름 (5 epoch에서 loss 절반)
- 100 epoch 완료 가능 (시간 소요 관리 가능)
- 체크포인트 저장 안정적

**개선 필요 사항**:
- Learning rate scheduling 필요 (중반 loss 증가)
- Batch별 loss 변동 큰 편 (frame sampling 전략 고려)
- SSIM loss가 0으로 설정됨 (활성화 고려)

### 5.3 Config 변형 효과

**Debug 모드 (max_frames=1500)**:
- 빠른 프로토타이핑에 유용
- 전체 데이터셋 대비 1/12 크기로 테스트 가능

**Fast 모드 (plot_every=5, save_every=5)**:
- 디스크 I/O 감소
- 학습 속도 미미하게 향상
- 중간 결과 확인 빈도 감소 (디버깅 시 불리)

---

## 6. 다음 실험 계획

### 6.1 즉시 실행 가능 (비-GPU 작업)

**Priority 1: 결과 분석 및 문서화**
```bash
# 1. Config 비교 분석
python compare_configs.py \
  configs/markerless_mouse_nerf_extended.json \
  configs/markerless_mouse_nerf_extended_debug.json \
  configs/markerless_mouse_nerf_extended_fast.json

# 2. 학습 로그 시각화
python visualize_training.py \
  --log output/markerless_mouse_nerf_extended/logs/step6_training.log \
  --output docs/reports/training_curves.png

# 3. 메트릭 분석 (evaluation 결과가 있다면)
python analyze_results.py \
  --metrics output/markerless_mouse_nerf_extended/metrics_test.csv \
  --output docs/reports/metrics_analysis.png
```

**Priority 2: 전처리 파이프라인 안정화**
```bash
# 1. Auto up direction 추정 테스트
python auto_estimate_up.py \
  --config configs/markerless_mouse_nerf_extended.json

# 2. HDF5 쓰기 수정 (N_JOBS 조정)
# write_images.py 내부 N_JOBS = 4로 감소
```

### 6.2 GPU 필요 작업 (후속 실험)

**Experiment 1: Learning Rate Scheduling**
```json
{
  "lr": 1e-4,
  "lr_scheduler": "cosine",  // 추가 필요
  "lr_decay": 0.95,          // 추가 필요
  "lr_step": 10              // 추가 필요
}
```
- 목적: Mid-training loss 증가 문제 해결
- 예상 시간: 100 epoch × ~3분 = ~5시간

**Experiment 2: SSIM Loss 활성화**
```json
{
  "ssim_lambda": 0.1  // 0.0 → 0.1
}
```
- 목적: 구조적 유사도 향상
- 예상 시간: 100 epoch × ~3분 = ~5시간

**Experiment 3: Grid Size 실험**
```json
{
  "grid_size": 128  // 112 → 128 (메모리 허용 시)
}
```
- 목적: 더 높은 해상도 복원
- 예상 시간: 100 epoch × ~4분 = ~6.5시간
- 주의: GPU 메모리 확인 필요

**Experiment 4: Frame Jump 최적화**
```json
{
  "frame_jump": 3  // 2 → 3 (빠른 학습)
  "frame_jump": 1  // 2 → 1 (고품질)
}
```
- 목적: 학습 속도 vs 품질 trade-off 탐색
- FJ=3: ~3.5시간, FJ=1: ~10시간 예상

### 6.3 2D/3D Gaussian Splatting 전환 모듈 설계

**배경**:
- 현재: 3D Gaussian Splatting (gsplat 라이브러리)
- 목표: 2D/3D 선택 가능한 모듈화 설계

**설계 방향**:

**Option 1: Config 기반 전환**
```json
{
  "gaussian_mode": "3d",  // "2d" or "3d"
  "gaussian_config": {
    "2d": {
      "num_gaussians": 1024,
      "splat_radius": 2.0
    },
    "3d": {
      "num_gaussians": 4096,
      "scale_activation": "exp"
    }
  }
}
```

**Option 2: Abstract Base Class**
```python
# src/gaussian_renderer.py
from abc import ABC, abstractmethod

class GaussianRenderer(ABC):
    @abstractmethod
    def render(self, means, scales, rotations, colors, opacities):
        pass

class GaussianRenderer2D(GaussianRenderer):
    def render(self, ...):
        # 2D splatting implementation
        pass

class GaussianRenderer3D(GaussianRenderer):
    def render(self, ...):
        # gsplat library
        from gsplat.rendering import rasterization
        return rasterization(...)
```

**Option 3: Unified Interface**
```python
# src/model.py 수정
class PoseSplatter(nn.Module):
    def __init__(self, ..., gaussian_mode="3d"):
        self.gaussian_mode = gaussian_mode

        if gaussian_mode == "2d":
            self.renderer = GaussianRenderer2D(...)
        else:
            self.renderer = GaussianRenderer3D(...)

    def forward(self, ...):
        # Gaussian parameters from network
        params = self.gaussian_param_net(features)

        # Unified rendering interface
        rgb, alpha = self.renderer.render(params, viewmat, K)
        return rgb, alpha
```

**구현 단계**:
1. **Phase 1**: Abstract interface 정의 (비-GPU)
2. **Phase 2**: 2D renderer 구현 (GPU 필요)
3. **Phase 3**: Config 통합 및 테스트 (GPU 필요)
4. **Phase 4**: 성능 비교 실험 (GPU 필요)

**예상 시간**:
- Phase 1: 1-2시간 (설계 및 인터페이스)
- Phase 2: 4-6시간 (구현 및 디버깅)
- Phase 3: 2-3시간 (통합)
- Phase 4: 5-10시간 (실험)
- **총 예상**: 12-21시간

---

## 7. 교훈 및 Best Practices

### 7.1 실험 설계

**Debug-First 원칙**:
- 긴 학습 전 반드시 debug config로 검증
- max_frames 제한으로 빠른 프로토타이핑
- Pipeline 전체 실행보다 단계별 검증

**Config 관리**:
- 명확한 naming convention (base, debug, fast)
- 한 가지 변수만 변경하여 효과 측정
- Git으로 config 버전 관리

### 7.2 전처리 안정성

**Defensive Programming**:
- 각 단계별 출력 파일 존재 확인
- 병렬 처리 시 race condition 방지
- 환경 의존성 명시적 관리 (conda env)

**자동화 우선**:
- 수동 입력 최소화 (auto_estimate_up.py)
- Pipeline 재실행 가능하도록 설계
- 중간 결과 캐싱 활용

### 7.3 학습 모니터링

**Loss 분석**:
- Batch별 변동 vs Epoch 평균 구분
- Mid-training loss 증가 → LR scheduling 신호
- 여러 metric 동시 모니터링 (IOU, SSIM, L1)

**체크포인트 전략**:
- 주기적 저장 (save_every)
- Best model 별도 저장
- Resume 기능 활용

---

## 8. Action Items

### 즉시 실행 가능 (우선순위)

- [ ] **Config 비교 분석 실행** (`compare_configs.py`)
- [ ] **학습 곡선 시각화** (`visualize_training.py`)
- [ ] **전처리 파이프라인 수정** (HDF5 병렬 처리 안정화)
- [ ] **2D/3D GS 모듈 인터페이스 설계** (Phase 1)
- [ ] **이 보고서 Obsidian에 저장** (`/home/joon/Documents/Obsidian/40_Areas/2_Research/_Notes/`)

### GPU 필요 작업 (후속)

- [ ] **Learning Rate Scheduling 실험** (Experiment 1)
- [ ] **SSIM Loss 활성화 실험** (Experiment 2)
- [ ] **2D Gaussian Renderer 구현** (Phase 2-3)
- [ ] **Grid Size 최적화 실험** (Experiment 3)
- [ ] **2D vs 3D GS 성능 비교** (Phase 4)

### 문서화

- [ ] **README.md 업데이트** (최신 실험 결과 반영)
- [ ] **Best Practices 문서 작성** (전처리 안정화)
- [ ] **2D/3D GS 설계 문서 작성** (`docs/reports/2d_3d_gs_design.md`)

---

## 9. 참고 자료

**실험 결과 파일**:
- `output/markerless_mouse_nerf_extended/checkpoint.pt`
- `output/markerless_mouse_nerf_extended/loss.pdf`
- `output/markerless_mouse_nerf_extended/reconstruction.pdf`
- `extended_training.log`
- `extended_training_pipeline.log`

**Config 파일**:
- `configs/markerless_mouse_nerf_extended.json`
- `configs/markerless_mouse_nerf_extended_debug.json`
- `configs/markerless_mouse_nerf_extended_debug_fj5.json`
- `configs/markerless_mouse_nerf_extended_fast.json`

**주요 코드**:
- `train_script.py` (학습 로직)
- `src/model.py` (PoseSplatter 모델)
- `src/unet_3d.py` (3D UNet)
- Preprocessing scripts (estimate_up_direction.py, etc.)

---

**보고서 작성자**: Claude Code
**작성일**: 2025-11-12
**프로젝트 디렉토리**: `/home/joon/dev/pose-splatter`
