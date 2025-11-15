# 2D vs 3D Gaussian Splatting 실험 Quick Start Guide

**작성일**: 2025-11-15
**목적**: 다음 세션에서 즉시 실험 시작할 수 있도록 준비

---

## TL;DR - 지금 바로 실행하기

### Option 1: 자동화 스크립트 (권장)

```bash
# Phase 1: Debug Mode (40-60분)
cd /home/joon/dev/pose-splatter
bash scripts/run_2d_3d_comparison.sh --phase1

# Phase 2: Short Training (4-6시간, Phase 1 성공 후)
bash scripts/run_2d_3d_comparison.sh --phase2

# 결과 분석
python3 scripts/analyze_results.py \
  --log2d output/2d_debug_*.log \
  --log3d output/3d_debug_*.log
```

### Option 2: 수동 실행

```bash
# 1. 환경 확인
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. 2D Debug (10 epochs, ~30분)
cd /home/joon/dev/pose-splatter
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_script.py configs/2d_3d_comparison_2d_debug.json --epochs 10

# 3. 3D Debug (10 epochs, ~30분)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_script.py configs/2d_3d_comparison_3d_debug.json --epochs 10
```

---

## 현재 상태 (2025-11-15)

### ✅ 완료된 작업

1. **환경 설정**:
   - Python 3.10.12 ✓
   - PyTorch 2.9.0 + CUDA ✓
   - Numpy 1.24.4 ✓
   - Dependencies 설치 완료 ✓

2. **코드 구현** (2025-11-12):
   - 2D/3D Gaussian Renderer 통합 ✓
   - 24개 단위 테스트 100% 통과 ✓
   - Config 기반 모드 전환 ✓

3. **실험 준비**:
   - GPU 메모리 최적화 Config 작성 ✓
   - grid_size: 64 → 48 (OOM 방지) ✓
   - max_frames: 50 → 30 (메모리 절감) ✓

4. **자동화**:
   - 실험 자동화 스크립트 ✓
   - 결과 분석 스크립트 ✓
   - 실험 계획 문서 ✓

### ⏳ 대기 중

1. **Phase 1: Debug Mode** (30-60분):
   - [ ] 2D debug (10 epochs)
   - [ ] 3D debug (10 epochs)

2. **Phase 2: Short Training** (4-6시간):
   - [ ] Config 생성 (2d_short, 3d_short)
   - [ ] 2D short (50 epochs)
   - [ ] 3D short (50 epochs)

3. **결과 분석**:
   - [ ] Metrics 비교
   - [ ] Visualization
   - [ ] 보고서 작성

---

## 빠른 시작 체크리스트

### 시작 전 (1분)

```bash
# GPU 확인
nvidia-smi

# 환경 확인
cd /home/joon/dev/pose-splatter
python3 -c "import torch, numpy; print('✓ Ready')"

# Config 확인
cat configs/2d_3d_comparison_2d_debug.json | grep -E "(grid_size|max_frames|gaussian_mode)"
```

**기대 결과**:
- GPU 사용 가능 (11.75 GB total)
- Python imports 성공
- grid_size: 48, max_frames: 30, gaussian_mode: 2d

### Phase 1 실행 (40-60분)

**자동 실행**:
```bash
cd /home/joon/dev/pose-splatter
bash scripts/run_2d_3d_comparison.sh --phase1
```

**수동 실행** (더 세밀한 제어):
```bash
# 2D Debug
cd /home/joon/dev/pose-splatter
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 train_script.py configs/2d_3d_comparison_2d_debug.json --epochs 10 \
  | tee output/2d_debug_manual.log

# 성공 확인
ls -lh output/2d_3d_comparison_2d_debug/checkpoint.pt

# 3D Debug
python3 train_script.py configs/2d_3d_comparison_3d_debug.json --epochs 10 \
  | tee output/3d_debug_manual.log

# 성공 확인
ls -lh output/2d_3d_comparison_3d_debug/checkpoint.pt
```

**모니터링**:
```bash
# GPU 메모리 (별도 터미널)
watch -n 1 nvidia-smi

# 학습 진행 (별도 터미널)
tail -f output/*.log
```

### Phase 1 검증 (5분)

```bash
# Loss 확인
grep "epoch loss" output/2d_debug_*.log | tail -3
grep "epoch loss" output/3d_debug_*.log | tail -3

# Checkpoint 확인
ls -lh output/2d_3d_comparison_2d_debug/checkpoint.pt
ls -lh output/2d_3d_comparison_3d_debug/checkpoint.pt

# GPU 메모리 peak 확인 (로그 확인)
# 예상: < 6GB
```

**성공 조건**:
- Loss 감소 추세 확인
- Checkpoint 파일 생성 (각 ~100-200 MB)
- GPU OOM 없음

---

## 문제 해결 (Troubleshooting)

### 1. CUDA Out of Memory

**증상**:
```
torch.OutOfMemoryError: CUDA out of memory
```

**해결**:
```bash
# grid_size 추가 감소
# configs/*.json 파일에서:
"grid_size": 48 → 32

# max_frames 추가 감소
"max_frames": 30 → 20

# 재시작
python3 train_script.py configs/...
```

### 2. Import Error (numpy)

**증상**:
```
ModuleNotFoundError: No module named 'numpy'
```

**해결**:
```bash
pip3 install numpy==1.24.4
python3 -c "import numpy; print('OK')"
```

### 3. 데이터 로딩 실패

**증상**:
```
FileNotFoundError: images/...
```

**해결**:
```bash
# 이전 학습 결과에서 복사
cp -r output/markerless_mouse_nerf_extended/images/* \
      output/2d_3d_comparison_2d_debug/images/
```

### 4. 학습이 매우 느림

**확인**:
```bash
# GPU 사용 확인
nvidia-smi
# GPU Utilization이 0%라면 CPU로 실행 중

# CUDA 사용 확인
python3 -c "import torch; print(torch.cuda.is_available())"
```

**해결**:
- 환경 변수 설정: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- PyTorch CUDA 버전 확인

---

## 실험 실행 타임라인

### Day 1 - Phase 1 (오늘)

| 시간 | 작업 | 소요 | 누적 |
|------|------|------|------|
| 10:00 | 환경 확인 | 5분 | 5분 |
| 10:05 | 2D Debug 시작 | 25분 | 30분 |
| 10:30 | 2D 결과 검증 | 5분 | 35분 |
| 10:35 | 3D Debug 시작 | 25분 | 60분 |
| 11:00 | 3D 결과 검증 | 5분 | 65분 |
| 11:05 | Phase 1 분석 | 10분 | 75분 |

**총 소요: ~1시간**

### Day 1 - Phase 2 준비 (오후)

| 시간 | 작업 | 소요 |
|------|------|------|
| 13:00 | Phase 2 Config 작성 | 15분 |
| 13:15 | 2D Short 시작 (백그라운드) | 2.5시간 |
| 15:45 | 2D 결과 확인 | 10분 |
| 16:00 | 3D Short 시작 (백그라운드) | 2.5시간 |
| 18:30 | 3D 결과 확인 | 10분 |
| 18:40 | 결과 분석 | 20분 |

**총 소요: ~5.5시간** (대부분 백그라운드)

---

## Next Steps

### Phase 1 성공 후

1. **결과 검토**:
   ```bash
   # Loss curves 확인
   grep "epoch loss" output/*_debug_*.log

   # Checkpoint 크기
   ls -lh output/*/checkpoint.pt
   ```

2. **Phase 2 Config 생성**:
   ```bash
   # 2d_short.json
   cp configs/2d_3d_comparison_2d_debug.json \
      configs/2d_3d_comparison_2d_short.json

   # Edit:
   # - max_frames: 30 → 100
   # - project_directory: .../2d_short/
   ```

3. **Phase 2 실행**:
   ```bash
   bash scripts/run_2d_3d_comparison.sh --phase2
   ```

### Phase 2 성공 후

1. **분석**:
   ```bash
   python3 scripts/analyze_results.py \
     --log2d output/2d_short_*.log \
     --log3d output/3d_short_*.log \
     --output output/comparison_analysis
   ```

2. **시각화 확인**:
   ```bash
   ls output/comparison_analysis/
   # - loss_convergence.png
   # - metrics_comparison.png
   # - comparison_table.md
   ```

3. **보고서 작성**:
   - `docs/reports/251115_2d_3d_experiment_results.md`
   - Quantitative + Qualitative analysis
   - Recommendation

---

## 빠른 명령어 참조

### 실험 실행

```bash
# Phase 1 자동
bash scripts/run_2d_3d_comparison.sh --phase1

# Phase 2 자동
bash scripts/run_2d_3d_comparison.sh --phase2

# 수동 (2D)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_script.py configs/2d_3d_comparison_2d_debug.json --epochs 10

# 수동 (3D)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_script.py configs/2d_3d_comparison_3d_debug.json --epochs 10
```

### 모니터링

```bash
# GPU
watch -n 1 nvidia-smi

# 로그
tail -f output/*.log

# Loss
grep "epoch loss" output/*.log | tail -10

# Progress
ps aux | grep train_script
```

### 분석

```bash
# 자동 분석
python3 scripts/analyze_results.py \
  --log2d output/2d_*.log \
  --log3d output/3d_*.log

# 수동 확인
grep "epoch loss" output/*.log
ls -lh output/*/checkpoint.pt
```

---

## 예상 결과

### Phase 1 Debug (10 epochs)

**2D Mode**:
- Training time: ~25 minutes
- GPU memory: ~3-4 GB
- Final loss: ~2-3 (초기값에서 감소)
- Checkpoint size: ~100-150 MB

**3D Mode**:
- Training time: ~25 minutes
- GPU memory: ~4-5 GB
- Final loss: ~2-3 (초기값에서 감소)
- Checkpoint size: ~150-200 MB

### Phase 2 Short (50 epochs)

**2D Mode**:
- Training time: ~2-3 hours
- GPU memory: ~4-5 GB
- Final loss: ~0.5-1.0
- IoU: > 0.7, PSNR: > 20 dB

**3D Mode**:
- Training time: ~2-3 hours
- GPU memory: ~5-6 GB
- Final loss: ~0.5-1.0 (가설: 더 낮음)
- IoU: > 0.7, PSNR: > 20 dB (가설: 더 높음)

---

## 중요 파일 위치

### Configs
- `configs/2d_3d_comparison_2d_debug.json` (Phase 1, 2D)
- `configs/2d_3d_comparison_3d_debug.json` (Phase 1, 3D)

### Scripts
- `scripts/run_2d_3d_comparison.sh` (자동화)
- `scripts/analyze_results.py` (결과 분석)
- `train_script.py` (학습 메인)

### Outputs
- `output/2d_3d_comparison_2d_debug/` (2D 결과)
- `output/2d_3d_comparison_3d_debug/` (3D 결과)
- `output/*_debug_*.log` (학습 로그)

### Docs
- `docs/reports/251115_2d_3d_comparison_experiment_plan.md` (실험 계획)
- `docs/reports/251112_2d_3d_renderer_implementation.md` (구현 상세)
- `docs/reports/251115_quick_start_guide.md` (이 문서)

---

## 마지막 체크

실험 시작 전 확인:

- [ ] GPU 사용 가능 (nvidia-smi)
- [ ] Python 환경 정상 (numpy, torch)
- [ ] Config 파일 준비 완료 (grid_size=48, max_frames=30)
- [ ] 디스크 공간 충분 (checkpoint ~200MB each)
- [ ] 시간 확보 (Phase 1: 1시간, Phase 2: 5시간)

**준비 완료! 다음 세션에서 즉시 시작 가능**

---

**작성자**: Claude Code
**작성일**: 2025-11-15
**Status**: Ready to Execute
**Command**: `bash scripts/run_2d_3d_comparison.sh --phase1`
