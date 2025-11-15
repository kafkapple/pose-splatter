# 세션 재개 가이드 (2025-11-15)

## 현재 완료 상태

### ✅ 완료된 작업

1. **2D/3D Gaussian Splatting 통합**
   - `src/gaussian_renderer.py`: 2D/3D 렌더러 구현 (584줄)
   - `src/model.py`: PoseSplatter 모델에 통합 (~200줄 수정)
   - Abstract base class 패턴으로 확장 가능한 구조

2. **Device Mismatch 수정**
   - `src/shape_carver.py`:
     - Line 110-111: `sample_nearest_pixels_torch()` - `.to(device)` 추가
     - Line 280: `compute_voxel_colors_torch()` - projected_coords device 수정
     - Line 287: sampled_colors device 수정
   - `src/model.py`:
     - Line 222, 245: logit_opacities shape 수정 (`.unsqueeze(-1)`)
     - Line 280: p_3d device 변환 추가
   - `src/gaussian_renderer.py`:
     - Line 193: opacity shape 수정 (`.squeeze(-1)`)
     - Line 196: gsplat API 3-value return 처리

3. **테스트 검증**
   - Integration tests: 4/4 통과
     - test_model_3d_mode: ✅
     - test_model_2d_mode: ✅
     - test_parameter_count: ✅
     - test_background_color: ✅
   - Checkpoint tests: 2/2 통과
     - 3D mode with checkpoint: ✅
     - 2D mode with checkpoint: ✅

4. **학습 환경 준비**
   - Config 파일 생성:
     - `configs/2d_3d_comparison_3d_debug.json`
     - `configs/2d_3d_comparison_2d_debug.json`
   - 출력 디렉토리 준비:
     - `output/2d_3d_comparison_3d_debug/`
     - `output/2d_3d_comparison_2d_debug/`
   - 데이터 복사 완료 (camera_params, zarr dataset 등)

5. **train_script.py 수정**
   - Line 333-334: gaussian_mode, gaussian_config 파라미터 추가
   - Line 338-341: 렌더러 타입 로깅 추가

### ⚠️ 발견된 문제점

1. **GPU 메모리 부족**
   - 문제: quaternion 변환에서 CUDA OOM (8.19 GiB 요청, 1.73 GiB 사용 가능)
   - 해결: grid_size를 112 → 64로 감소 (이미 config 수정 완료)
   - 추가 옵션: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

2. **Python 환경 문제**
   - 문제: numpy import 실패 (반복적 발생)
   - 원인: 환경 초기화 또는 패키지 충돌로 추정
   - 상태: 미해결 (다음 세션에서 재시도 필요)

## 다음 세션에서 할 일

### 1단계: 환경 검증 (5분)

```bash
# GPU 상태 확인
nvidia-smi

# Python 패키지 확인
python3 -c "import numpy; import torch; import gsplat; print('✓ All packages OK')"

# 필요시 재설치
pip install numpy==1.24.3 torch-scatter
```

### 2단계: 2D/3D 비교 학습 실행 (30-60분)

#### Option A: 순차 실행 (안전)

```bash
cd /home/joon/dev/pose-splatter

# 3D 모드 학습 (baseline)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 train_script.py configs/2d_3d_comparison_3d_debug.json \
> output/3d_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 학습 상태 모니터링
tail -f output/3d_training_*.log

# 3D 완료 후 2D 시작
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 train_script.py configs/2d_3d_comparison_2d_debug.json \
> output/2d_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Option B: 병렬 실행 (빠름, 메모리 주의)

```bash
# GPU 0에서 3D
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 train_script.py configs/2d_3d_comparison_3d_debug.json \
> output/3d_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 동시에 2D도 시작 (메모리 충분하면)
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 train_script.py configs/2d_3d_comparison_2d_debug.json \
> output/2d_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 3단계: 결과 비교 및 시각화 (20분)

```bash
# 학습 완료 확인
ls -lh output/2d_3d_comparison_*/checkpoint.pt

# Loss 그래프 비교
ls -lh output/2d_3d_comparison_*/loss.pdf

# 재구성 결과 비교
ls -lh output/2d_3d_comparison_*/reconstruction.pdf

# 이미지 출력 비교
ls -lh output/2d_3d_comparison_*/images/
```

### 4단계: 정량적 비교 분석

생성할 메트릭:
- Training loss curves (2D vs 3D)
- Validation PSNR
- Rendering time per frame
- GPU memory usage
- Number of Gaussians per mode

## 예상 학습 시간

**Config 설정:**
- max_frames: 50
- valid_every: 5
- epochs: 50
- grid_size: 64 (메모리 최적화)

**예상 시간:**
- 3D 모드: ~20-30분 (50 프레임 × 50 에포크)
- 2D 모드: ~15-25분 (2D가 더 빠를 것으로 예상)

## 문제 해결 가이드

### GPU OOM 발생 시

1. **grid_size 더 줄이기**
   ```json
   "grid_size": 48  // 64 → 48
   ```

2. **max_frames 줄이기**
   ```json
   "max_frames": 30  // 50 → 30
   ```

3. **batch size 확인**
   - train_script.py에서 배치 크기 확인
   - DataLoader workers 수 조정

### Import 오류 발생 시

```bash
# 가상환경 확인
which python3
pip list | head -20

# numpy 강제 재설치
pip uninstall -y numpy
pip install numpy==1.24.3

# torch 호환성 확인
python3 -c "import torch; print(torch.__version__)"
```

### 데이터 파일 없음 오류 시

```bash
# 필요 파일 복사
cp output/markerless_mouse_nerf_extended/camera_params.h5 output/2d_3d_comparison_*/
cp output/markerless_mouse_nerf_extended/*.npz output/2d_3d_comparison_*/
cp output/markerless_mouse_nerf_extended/volume_sum.npy output/2d_3d_comparison_*/
cp -r output/markerless_mouse_nerf_extended/images/images.zarr output/2d_3d_comparison_*/images/
```

## 주요 파일 위치

### 코드
- `src/gaussian_renderer.py` - 2D/3D 렌더러 구현
- `src/model.py` - PoseSplatter 통합
- `train_script.py` - 학습 스크립트
- `src/shape_carver.py` - Volume carving (device fixes)

### Config
- `configs/2d_3d_comparison_3d_debug.json` - 3D 모드
- `configs/2d_3d_comparison_2d_debug.json` - 2D 모드

### 테스트
- `tests/test_model_integration.py` - 통합 테스트
- `tests/test_renderer_simple.py` - 렌더러 단순 테스트
- `tests/test_with_checkpoint.py` - Checkpoint 검증

### 출력
- `output/2d_3d_comparison_3d_debug/` - 3D 결과
- `output/2d_3d_comparison_2d_debug/` - 2D 결과
- `output/3d_training_*.log` - 학습 로그
- `output/2d_training_*.log` - 학습 로그

## 성공 기준

### 최소 목표
- [x] 2D/3D 렌더러 구현 및 통합
- [x] 모든 테스트 통과
- [ ] 2D/3D 각각 최소 10 에포크 학습 성공
- [ ] Loss curve 생성 및 비교

### 최적 목표
- [ ] 50 에포크 완전 학습
- [ ] PSNR/SSIM 정량적 비교
- [ ] 렌더링 속도 벤치마크
- [ ] 시각적 품질 비교 (reconstruction.pdf)

## 다음 단계 (학습 완료 후)

1. **Monocular 3D Prior 통합**
   - MAMMAL mouse mesh 활용
   - `fit_monocular.py` 통합
   - 단일 뷰에서 3D 재구성

2. **성능 최적화**
   - 2D 렌더러 CUDA 커널 작성
   - Batch 처리 최적화
   - 메모리 효율성 개선

3. **전체 데이터셋 학습**
   - max_frames: None (모든 프레임)
   - epochs: 100+
   - grid_size: 112 (원래 크기)

## 참고 문서

- `docs/reports/251112_2d_3d_renderer_implementation.md` - 구현 상세
- `docs/reports/251114_monocular_3d_prior_integration_plan.md` - 모노큘러 계획
- `docs/reports/2d_3d_gs_design.md` - 설계 문서

## 명령어 빠른 참조

```bash
# 상태 확인
ps aux | grep train_script.py
nvidia-smi
tail -50 output/*_training_*.log

# 프로세스 종료
pkill -f train_script.py

# 로그 모니터링
watch -n 5 tail -30 output/3d_training_*.log

# 결과 확인
ls -lhR output/2d_3d_comparison_*/
```

---

**작성자**: Claude Code
**날짜**: 2025-11-15
**세션**: 2D/3D Gaussian Splatting 통합 완료
