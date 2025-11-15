# 2D vs 3D Gaussian Splatting 비교 실험 계획

**날짜**: 2025-11-15
**프로젝트**: pose-splatter
**작업**: 2D/3D Gaussian Renderer 성능 비교 실험

---

## Executive Summary

이전 세션에서 완료한 2D/3D Gaussian Splatting 통합 구현을 바탕으로, 두 렌더링 방식의 성능과 품질을 비교하는 체계적인 실험을 진행합니다.

**핵심 목표**:
- 2D vs 3D Gaussian Splatting 렌더링 품질 비교
- 학습 속도 및 GPU 메모리 사용량 비교
- 실제 mouse pose reconstruction 작업에서의 적합성 평가

---

## 1. 이전 작업 요약

### 1.1 완료된 구현 (2025-11-12)

✅ **2D/3D Gaussian Renderer 통합**:
- Abstract renderer interface
- GaussianRenderer2D (9 params/Gaussian)
- GaussianRenderer3D (14 params/Gaussian)
- Config 기반 모드 전환
- 24개 단위/통합 테스트 100% 통과

### 1.2 이전 실험 결과 (2025-11-15 01:49)

❌ **3D Debug 학습 실패**:
```
Config: 2d_3d_comparison_3d_debug.json
Grid size: 64
Max frames: 50
Dataset: 3000 samples

Error: CUDA Out of Memory
- Attempted allocation: 8.19 GiB
- GPU capacity: 11.75 GiB
- Free memory: 1.73 GiB
- Process memory: 9.32 GiB
```

**근본 원인**:
- grid_size=64 → 64³ = 262,144 voxels
- 각 voxel당 14 parameters (3D mode)
- Batch processing 시 메모리 폭증
- quaternion conversion (torch.linalg.eigh) 추가 메모리

### 1.3 메모리 최적화 조치

✅ **Config 수정 완료**:
```json
{
  "grid_size": 48,      // 64 → 48 (64% 메모리 감소)
  "max_frames": 30,     // 50 → 30 (40% 감소)
  "gaussian_mode": "2d" or "3d"
}
```

**예상 메모리**:
- grid_size=64: ~8-10 GB (❌ OOM)
- grid_size=48: ~3-4 GB (✅ Safe)
- grid_size=32: ~1-2 GB (✅ Very Safe, 품질 저하)

**Trade-off**:
| Grid Size | Memory | Voxels | Quality | Status |
|-----------|--------|---------|---------|--------|
| 64 | 8-10 GB | 262K | Highest | ❌ OOM |
| 48 | 3-4 GB | 110K | Good | ✅ Target |
| 32 | 1-2 GB | 33K | Lower | Fallback |

---

## 2. 실험 설계

### 2.1 실험 목적

**Primary Questions**:
1. 2D vs 3D Gaussian Splatting 중 어느 것이 mouse reconstruction에 더 적합한가?
2. 학습 속도, 메모리, 품질 trade-off는?
3. Production에서 사용할 최적 설정은?

**Evaluation Metrics**:
- **품질**: IoU (mask), PSNR/SSIM (RGB)
- **속도**: Training time per epoch, inference time
- **메모리**: Peak GPU memory, 평균 사용량
- **안정성**: Loss convergence, gradient stability

### 2.2 실험 단계 (Debug-First 원칙)

**⚠️ 중요**: 30분 이상 소요 예상 실험은 반드시 debug 모드 먼저 실행

#### Phase 1: Debug Mode (PoC 검증) [30-60분]

**목적**: Config 검증 및 OOM 방지

**Config**:
- `2d_3d_comparison_2d_debug.json`
- `2d_3d_comparison_3d_debug.json`

**설정**:
```json
{
  "grid_size": 48,
  "max_frames": 30,
  "epochs": 10,        // Quick validation
  "valid_every": 2,
  "save_every": 2
}
```

**검증 항목**:
- [ ] 데이터 로딩 성공
- [ ] 모델 생성 성공 (2D: 9 params, 3D: 14 params)
- [ ] Forward pass 성공 (RGB + alpha)
- [ ] Backward pass 성공 (gradient)
- [ ] Loss 감소 추세 확인
- [ ] Checkpoint 저장 성공
- [ ] GPU 메모리 < 6GB
- [ ] 1 epoch 소요 시간 측정

**예상 시간**:
- 2D mode: 10 epochs × ~2-3 min/epoch = 20-30분
- 3D mode: 10 epochs × ~2-3 min/epoch = 20-30분
- Total: 40-60분

**실패 조건**:
- OOM 발생 → grid_size 48 → 32로 감소
- Loss NaN → Learning rate 조정
- 데이터 로딩 실패 → 데이터셋 재생성

#### Phase 2: Short Training (성능 비교) [2-4시간]

**목적**: 초기 수렴 속도 및 품질 비교

**조건**: Phase 1 debug 모두 성공

**Config**:
```json
{
  "grid_size": 48,
  "max_frames": 100,     // 30 → 100
  "epochs": 50,          // 10 → 50
  "valid_every": 5,
  "save_every": 5
}
```

**Metrics 수집**:
- Loss curves (training + validation)
- IoU progression
- PSNR/SSIM progression
- GPU memory usage (peak + avg)
- Training time per epoch
- Checkpoint size

**비교 분석**:
- 2D vs 3D loss convergence
- Quality metrics at epoch 50
- Memory efficiency
- Speed efficiency

**예상 시간**:
- 2D mode: 50 epochs × ~3 min/epoch = 2.5시간
- 3D mode: 50 epochs × ~3 min/epoch = 2.5시간
- Total: 5시간 (병렬 실행 불가 시)

#### Phase 3: Full Training (최종 평가) [선택 사항, 10-20시간]

**목적**: Production-level 품질 평가

**조건**: Phase 2 결과 분석 후 결정

**Config**:
```json
{
  "grid_size": 64,       // Quality 우선
  "max_frames": 500,     // Full dataset
  "epochs": 200,         // Convergence
  "valid_every": 10,
  "save_every": 10
}
```

**주의**:
- grid_size=64는 OOM 위험
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 필수
- 장시간 학습 → nohup 백그라운드 실행

---

## 3. 실험 실행 가이드

### 3.1 사전 준비

**환경 확인**:
```bash
# Python & PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Numpy (중요!)
python3 -c "import numpy; print(f'Numpy: {numpy.__version__}')"

# Dependencies
pip3 list | grep -E "(torch|numpy|matplotlib|gsplat)"

# GPU 상태
nvidia-smi
```

**데이터 준비**:
```bash
# 데이터 디렉토리 확인
ls -lh /home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf/

# Output 디렉토리 확인
ls -lh /home/joon/dev/pose-splatter/output/2d_3d_comparison_*
```

**Config 검증**:
```bash
# 2D config
cat configs/2d_3d_comparison_2d_debug.json | grep -E "(grid_size|max_frames|gaussian_mode)"

# 3D config
cat configs/2d_3d_comparison_3d_debug.json | grep -E "(grid_size|max_frames|gaussian_mode)"
```

### 3.2 Phase 1 실행 (Debug Mode)

#### 3.2.1 2D Debug

```bash
# 수동 실행 (로그 확인 가능)
cd /home/joon/dev/pose-splatter && \
python3 train_script.py configs/2d_3d_comparison_2d_debug.json \
  --epochs 10

# 백그라운드 실행
cd /home/joon/dev/pose-splatter && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 train_script.py configs/2d_3d_comparison_2d_debug.json \
  --epochs 10 \
  > output/2d_debug_$(date +%Y%m%d_%H%M).log 2>&1 &

# 로그 모니터링
tail -f output/2d_debug_*.log
```

**검증**:
```bash
# Loss 확인
grep "epoch loss" output/2d_debug_*.log

# Checkpoint 확인
ls -lh output/2d_3d_comparison_2d_debug/checkpoint.pt

# GPU 메모리 확인 (학습 중)
watch -n 1 nvidia-smi
```

#### 3.2.2 3D Debug

```bash
# 수동 실행
cd /home/joon/dev/pose-splatter && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_script.py configs/2d_3d_comparison_3d_debug.json \
  --epochs 10

# 백그라운드 실행
cd /home/joon/dev/pose-splatter && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 train_script.py configs/2d_3d_comparison_3d_debug.json \
  --epochs 10 \
  > output/3d_debug_$(date +%Y%m%d_%H%M).log 2>&1 &

# 로그 모니터링
tail -f output/3d_debug_*.log
```

**중요**: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 필수!

### 3.3 Phase 2 실행 (Short Training)

**조건**: Phase 1 debug 성공 확인 후

#### 3.3.1 Config 생성

새로운 config 파일 생성:
- `configs/2d_3d_comparison_2d_short.json`
- `configs/2d_3d_comparison_3d_short.json`

```json
{
  // Base config와 동일
  "grid_size": 48,
  "max_frames": 100,
  "project_directory": "output/2d_3d_comparison_2d_short/",
  "gaussian_mode": "2d" // or "3d"
}
```

#### 3.3.2 실행

```bash
# 2D Short Training
cd /home/joon/dev/pose-splatter && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 train_script.py configs/2d_3d_comparison_2d_short.json \
  --epochs 50 \
  > output/2d_short_$(date +%Y%m%d_%H%M).log 2>&1 &

# 3D Short Training
cd /home/joon/dev/pose-splatter && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 train_script.py configs/2d_3d_comparison_3d_short.json \
  --epochs 50 \
  > output/3d_short_$(date +%Y%m%d_%H%M).log 2>&1 &

# 병렬 실행 불가 (GPU 메모리)
# 한 실험씩 순차 실행
```

### 3.4 실험 모니터링

**GPU 메모리**:
```bash
# 실시간 모니터링
watch -n 1 nvidia-smi

# Peak 메모리 기록
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > gpu_mem.csv
```

**Loss 추적**:
```bash
# Real-time loss
tail -f output/*.log | grep "epoch loss"

# Loss plot (학습 후)
python3 -c "
import matplotlib.pyplot as plt
import re

with open('output/2d_debug.log') as f:
    losses = [float(re.search('loss: ([\d.]+)', line).group(1))
              for line in f if 'epoch loss' in line]
plt.plot(losses)
plt.savefig('loss_curve.png')
"
```

**Progress**:
```bash
# Epoch 진행 상황
tail -100 output/*.log | grep -E "(epoch|%)"

# 예상 완료 시간 계산
# (총 epochs - 현재 epoch) × epoch당 시간
```

### 3.5 문제 해결

#### OOM 발생 시

```bash
# 1. 학습 중단
pkill -f train_script.py

# 2. grid_size 감소
# configs/*.json에서 grid_size: 48 → 32

# 3. max_frames 감소
# configs/*.json에서 max_frames: 30 → 20

# 4. 재시작
python3 train_script.py configs/...
```

#### Import 오류 시

```bash
# Numpy 재설치
pip3 uninstall -y numpy && pip3 install numpy==1.24.4

# Dependencies 확인
pip3 check
```

#### 데이터 없음 오류 시

```bash
# 데이터 복사 (이전 학습 결과 활용)
cp -r output/markerless_mouse_nerf_extended/images/* \
      output/2d_3d_comparison_2d_debug/images/

cp -r output/markerless_mouse_nerf_extended/images/* \
      output/2d_3d_comparison_3d_debug/images/
```

---

## 4. 결과 분석 계획

### 4.1 정량적 비교

**Metrics Table**:
| Metric | 2D Mode | 3D Mode | Winner |
|--------|---------|---------|--------|
| Final IoU | ? | ? | ? |
| Final PSNR | ? | ? | ? |
| Training Time | ? | ? | ? |
| Peak GPU Memory | ? | ? | ? |
| Params/Gaussian | 9 | 14 | 2D |
| Checkpoint Size | ? | ? | ? |

**Loss Curves**:
- Training loss (2D vs 3D overlay)
- Validation loss
- IoU progression
- PSNR progression

### 4.2 정성적 비교

**Visualization**:
- Rendered images (2D vs 3D vs Ground Truth)
- Mask predictions
- Failure cases
- Temporal consistency

**Quality Assessment**:
- Reconstruction fidelity
- Geometric accuracy
- Temporal smoothness
- Artifact analysis

### 4.3 분석 스크립트

```python
# analyze_results.py
import numpy as np
import matplotlib.pyplot as plt
import json

def compare_experiments(log_2d, log_3d):
    # Parse logs
    losses_2d = parse_log(log_2d)
    losses_3d = parse_log(log_3d)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    axes[0, 0].plot(losses_2d['iou'], label='2D')
    axes[0, 0].plot(losses_3d['iou'], label='3D')
    axes[0, 0].set_title('IoU Loss')
    axes[0, 0].legend()

    # Training time
    axes[0, 1].bar(['2D', '3D'],
                   [losses_2d['time'], losses_3d['time']])
    axes[0, 1].set_title('Training Time')

    # Memory usage
    axes[1, 0].bar(['2D', '3D'],
                   [losses_2d['mem'], losses_3d['mem']])
    axes[1, 0].set_title('Peak GPU Memory')

    # Final metrics
    metrics = ['IoU', 'PSNR']
    x = np.arange(len(metrics))
    axes[1, 1].bar(x - 0.2, [losses_2d['final_iou'], losses_2d['final_psnr']],
                   width=0.4, label='2D')
    axes[1, 1].bar(x + 0.2, [losses_3d['final_iou'], losses_3d['final_psnr']],
                   width=0.4, label='3D')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_title('Final Metrics')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('comparison.png')

    return {
        '2d': losses_2d,
        '3d': losses_3d
    }
```

---

## 5. 자동화 스크립트

### 5.1 전체 실험 자동화

```bash
#!/bin/bash
# run_2d_3d_comparison.sh

set -e  # Exit on error

PROJECT_ROOT="/home/joon/dev/pose-splatter"
cd $PROJECT_ROOT

# 환경 변수
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 날짜 태그
DATE_TAG=$(date +%Y%m%d_%H%M)

echo "=== 2D vs 3D Gaussian Splatting Comparison Experiment ==="
echo "Date: $(date)"
echo "Project: $PROJECT_ROOT"
echo ""

# Phase 1: Debug Mode (2D)
echo "========================================="
echo "Phase 1.1: 2D Debug Mode (10 epochs)"
echo "========================================="
LOG_2D_DEBUG="output/2d_debug_${DATE_TAG}.log"
python3 train_script.py configs/2d_3d_comparison_2d_debug.json \
  --epochs 10 \
  > $LOG_2D_DEBUG 2>&1

# 검증
if [ $? -eq 0 ]; then
  echo "✅ 2D Debug SUCCESS"
  grep "epoch loss" $LOG_2D_DEBUG | tail -3
else
  echo "❌ 2D Debug FAILED"
  tail -20 $LOG_2D_DEBUG
  exit 1
fi

echo ""

# Phase 1: Debug Mode (3D)
echo "========================================="
echo "Phase 1.2: 3D Debug Mode (10 epochs)"
echo "========================================="
LOG_3D_DEBUG="output/3d_debug_${DATE_TAG}.log"
python3 train_script.py configs/2d_3d_comparison_3d_debug.json \
  --epochs 10 \
  > $LOG_3D_DEBUG 2>&1

# 검증
if [ $? -eq 0 ]; then
  echo "✅ 3D Debug SUCCESS"
  grep "epoch loss" $LOG_3D_DEBUG | tail -3
else
  echo "❌ 3D Debug FAILED"
  tail -20 $LOG_3D_DEBUG
  exit 1
fi

echo ""
echo "========================================="
echo "Phase 1 Complete!"
echo "========================================="
echo "2D Log: $LOG_2D_DEBUG"
echo "3D Log: $LOG_3D_DEBUG"
echo ""
echo "Next: Review debug results, then run Phase 2"
echo "  bash run_2d_3d_comparison.sh --phase2"
```

### 5.2 Phase 2 자동화

```bash
#!/bin/bash
# run_phase2.sh

# Phase 2: Short Training (50 epochs)
# 조건: Phase 1 debug 성공

PROJECT_ROOT="/home/joon/dev/pose-splatter"
cd $PROJECT_ROOT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATE_TAG=$(date +%Y%m%d_%H%M)

echo "========================================="
echo "Phase 2: Short Training (50 epochs)"
echo "========================================="

# 2D Short Training
echo "2D Short Training..."
LOG_2D_SHORT="output/2d_short_${DATE_TAG}.log"
nohup python3 train_script.py configs/2d_3d_comparison_2d_short.json \
  --epochs 50 \
  > $LOG_2D_SHORT 2>&1 &
PID_2D=$!

echo "2D Training started (PID: $PID_2D)"
echo "Log: tail -f $LOG_2D_SHORT"
echo ""

# Wait for 2D to finish
wait $PID_2D
echo "✅ 2D Short Training Complete"
echo ""

# 3D Short Training
echo "3D Short Training..."
LOG_3D_SHORT="output/3d_short_${DATE_TAG}.log"
nohup python3 train_script.py configs/2d_3d_comparison_3d_short.json \
  --epochs 50 \
  > $LOG_3D_SHORT 2>&1 &
PID_3D=$!

echo "3D Training started (PID: $PID_3D)"
echo "Log: tail -f $LOG_3D_SHORT"
echo ""

# Wait for 3D to finish
wait $PID_3D
echo "✅ 3D Short Training Complete"
echo ""

echo "========================================="
echo "Phase 2 Complete!"
echo "========================================="
echo "2D Log: $LOG_2D_SHORT"
echo "3D Log: $LOG_3D_SHORT"
echo ""
echo "Next: Analyze results with analyze_results.py"
```

---

## 6. Expected Outcomes

### 6.1 성공 조건

**Debug Mode (Phase 1)**:
- [ ] 2D debug 10 epochs 완료 (20-30분)
- [ ] 3D debug 10 epochs 완료 (20-30분)
- [ ] GPU 메모리 < 6GB
- [ ] Loss 감소 추세 확인
- [ ] Checkpoint 저장 성공

**Short Training (Phase 2)**:
- [ ] 2D short 50 epochs 완료 (~2.5시간)
- [ ] 3D short 50 epochs 완료 (~2.5시간)
- [ ] IoU > 0.7, PSNR > 20 dB
- [ ] 안정적인 convergence
- [ ] Visualization 품질 양호

### 6.2 예상 결과

**가설**:
- **2D Mode**: 빠름, 메모리 효율, 품질 중간
- **3D Mode**: 느림, 메모리 많음, 품질 높음

**Trade-off**:
| Aspect | 2D Advantage | 3D Advantage |
|--------|-------------|-------------|
| Speed | ✅ Faster | Slower |
| Memory | ✅ Lower | Higher |
| Params | ✅ 9 vs 14 | More expressive |
| Quality | ? | ✅ ? (가설) |
| Depth | Implicit | ✅ Explicit |

### 6.3 의사결정 기준

**2D를 선택하는 경우**:
- 속도가 중요 (실시간 추론)
- 메모리 제약 (edge device)
- 품질 차이가 미미 (< 2% IoU/PSNR)

**3D를 선택하는 경우**:
- 품질이 최우선
- 3D consistency 필요
- 충분한 GPU 리소스

---

## 7. Timeline

### Day 1 (오늘)

**09:00 - 10:00**: 환경 설정 및 문서 작성 ✅
**10:00 - 11:00**: Phase 1 Debug (2D + 3D)
**11:00 - 11:30**: Debug 결과 분석 및 검증
**11:30 - 12:00**: Phase 2 Config 준비

### Day 1 (오후)

**13:00 - 15:30**: Phase 2 Short Training (2D)
**15:30 - 18:00**: Phase 2 Short Training (3D)
**18:00 - 19:00**: 결과 분석 및 시각화

### Day 2 (선택)

**Full Training**: Phase 2 결과에 따라 결정

---

## 8. Deliverables

### 8.1 실험 완료 시

**코드**:
- [x] 최적화된 configs (2D/3D debug/short)
- [ ] 자동화 스크립트 (run_experiment.sh)
- [ ] 분석 스크립트 (analyze_results.py)

**데이터**:
- [ ] 2D/3D checkpoints
- [ ] Training logs
- [ ] Loss curves
- [ ] Rendered images

**문서**:
- [x] 실험 계획서 (이 문서)
- [ ] 실험 결과 보고서
- [ ] 비교 분석 보고서
- [ ] Production 권장사항

### 8.2 보고서 구조 (예정)

**실험 결과 보고서** (251115_2d_3d_experiment_results.md):
1. Executive Summary
2. Experimental Setup
3. Quantitative Results
4. Qualitative Analysis
5. Discussion
6. Conclusion & Recommendations

---

## 9. Risk Mitigation

### 9.1 OOM 위험

**완화 조치**:
- ✅ grid_size: 64 → 48
- ✅ max_frames: 50 → 30
- ✅ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- Fallback: grid_size 32

### 9.2 학습 실패 위험

**완화 조치**:
- Debug-first 원칙
- Checkpoint 자동 저장
- Loss monitoring
- Early stopping

### 9.3 시간 초과 위험

**완화 조치**:
- 단계적 실험 (Debug → Short → Full)
- 병렬 실행 (불가능 시 순차)
- nohup 백그라운드 실행
- 진행 상황 모니터링

---

## 10. Next Steps

### 즉시 실행 (지금)

1. **자동화 스크립트 작성**:
   - `scripts/run_2d_3d_comparison.sh`
   - `scripts/analyze_results.py`

2. **Phase 1 실행**:
   - 2D debug (10 epochs, 30분)
   - 3D debug (10 epochs, 30분)

3. **검증**:
   - Loss 감소 확인
   - GPU 메모리 확인
   - Checkpoint 확인

### 다음 세션

4. **Phase 2 실행** (Phase 1 성공 시):
   - Short training configs 생성
   - 2D/3D short training (각 2.5시간)

5. **결과 분석**:
   - Metrics 비교
   - Visualization
   - 보고서 작성

---

## 11. References

**설계 문서**:
- `docs/reports/2d_3d_gs_design.md`
- `docs/reports/251112_2d_3d_renderer_implementation.md`

**코드**:
- `src/gaussian_renderer.py` - Renderer 구현
- `src/model.py` - PoseSplatter 통합
- `train_script.py` - 학습 스크립트

**Config**:
- `configs/2d_3d_comparison_2d_debug.json`
- `configs/2d_3d_comparison_3d_debug.json`

**가이드**:
- `/home/joon/CLAUDE.md` - 실험 실행 가이드
- `/home/joon/dev/CLAUDE.md` - PKM 연구 노트 규칙

---

**작성자**: Claude Code
**작성일**: 2025-11-15
**Status**: Ready to Execute
**Next**: Phase 1 Debug Mode
