# A6000 서버 빠른 시작 가이드

새로운 A6000 서버에서 프로젝트를 시작할 때 사용하는 체크리스트입니다.

## 1. 환경 문제 해결 (필수!)

```bash
# 프로젝트 디렉토리로 이동
cd /home/joon/pose-splatter  # 또는 /home/joon/dev/pose-splatter

# 환경 진단 및 자동 수정
bash scripts/utils/fix_environment.sh
```

**이 스크립트가 하는 일**:
- ✓ NumPy 버전 충돌 해결 (NumPy < 2.0 강제)
- ✓ torch_scatter 설치 확인 및 설치
- ✓ torchvision 재설치
- ✓ 모든 import 테스트

**예상 시간**: 2-3분

---

## 2. 데이터 확인

```bash
# 데이터 디렉토리 확인
ls -lh data/markerless_mouse_1_nerf/

# 필수 파일 확인
python scripts/utils/verify_mammal_data.py
```

**필요한 파일**:
- ✓ `new_cam.pkl` (카메라 파라미터)
- ✓ `videos_undist/0.mp4 ~ 5.mp4` (6개 카메라 영상)
- ✓ `simpleclick_undist/0.mp4 ~ 5.mp4` (마스크 영상)

---

## 3. 전처리 (한 번만)

```bash
# 전체 전처리 파이프라인 실행
bash scripts/preprocessing/run_full_preprocessing.sh \
  configs/baseline/markerless_mouse_nerf.json
```

**예상 시간**: 10-30분

**생성되는 파일**:
- `data/markerless_mouse_1_nerf/camera_params.h5`
- `data/markerless_mouse_1_nerf/vertical_lines.npz`
- `data/markerless_mouse_1_nerf/center_rotation.npz`
- `data/markerless_mouse_1_nerf/images/` (ZARR 형식)

---

## 4. 학습 실행

### 옵션 A: 2D GS (고품질, A6000 권장)

```bash
# 전경에서 실행 (테스트용)
bash scripts/training/run_training.sh \
  configs/templates/a6000_2d.json --epochs 50

# 백그라운드 실행 (실제 학습)
nohup bash scripts/training/run_training.sh \
  configs/templates/a6000_2d.json --epochs 50 \
  > training_2d.log 2>&1 &

# 진행 상황 모니터링
tail -f training_2d.log
```

**예상 시간**: 10-15시간 (50 epochs)
**GPU 메모리**: ~18-20GB
**예상 PSNR**: ~28-30 dB

### 옵션 B: 3D GS (빠른 테스트)

```bash
bash scripts/training/run_training.sh \
  configs/templates/rtx3060_3d.json --epochs 50
```

**예상 시간**: 6-8시간
**GPU 메모리**: ~6-8GB

### 옵션 C: 디버그 모드 (5-10분 빠른 검증)

```bash
bash scripts/training/run_training.sh \
  configs/templates/debug_quick.json --epochs 5
```

---

## 5. 모니터링

```bash
# GPU 사용량 확인
watch -n 2 nvidia-smi

# 로그 실시간 확인
tail -f training_2d.log

# 또는
tail -f output/a6000_2d/logs/step6_training.log
```

---

## 문제 해결

### 오류 1: NumPy 버전 충돌

**증상**:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.1
```

**해결**:
```bash
bash scripts/utils/fix_environment.sh
```

### 오류 2: torch_scatter 없음

**증상**:
```
ModuleNotFoundError: No module named 'torch_scatter'
```

**해결**:
```bash
bash scripts/utils/fix_environment.sh
```

### 오류 3: CUDA Out of Memory

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결**: 설정 파일 수정
```json
{
  "image_downsample": 4,  // 2 → 4 (해상도 감소)
  "grid_size": 112,       // 128 → 112 (복셀 수 감소)
  "gaussian_config": {
    "batch_size": 3       // 5 → 3 (2D GS만)
  }
}
```

---

## 완전 자동화 (전처리 + 학습)

```bash
# 1. 환경 수정
bash scripts/utils/fix_environment.sh

# 2. 전처리
bash scripts/preprocessing/run_full_preprocessing.sh \
  configs/baseline/markerless_mouse_nerf.json

# 3. 백그라운드 학습
nohup bash scripts/training/run_training.sh \
  configs/templates/a6000_2d.json --epochs 50 \
  > training.log 2>&1 &

# 4. 모니터링
tail -f training.log
```

---

## 결과 확인

학습 완료 후:

```bash
# 체크포인트 확인
ls -lh output/a6000_2d/checkpoint.pt

# 렌더링 결과
ls output/a6000_2d/renders/

# 메트릭 확인
cat output/a6000_2d/logs/step6_training.log | grep PSNR
```

---

## 참고 문서

- **전체 설정 가이드**: [docs/reports/CONFIGURATION_GUIDE.md](docs/reports/CONFIGURATION_GUIDE.md)
- **환경 문제 해결**: [docs/troubleshooting/environment_errors.md](docs/troubleshooting/environment_errors.md)
- **전처리 가이드**: [docs/reports/PREPROCESSING_GUIDE_BEGINNER.md](docs/reports/PREPROCESSING_GUIDE_BEGINNER.md)
- **템플릿 사용법**: [configs/templates/README.md](configs/templates/README.md)
