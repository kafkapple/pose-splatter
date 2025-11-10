# 안전한 시각화 실행 가이드

## ⚠️ 중요: GPU 메모리 관리

현재 환경에서는 백그라운드 프로세스가 GPU 메모리를 사용 중이므로, 시각화 실행 시 **반드시** 아래 단계를 따라야 합니다.

---

## 📊 실행 전 체크리스트

### 1단계: GPU 메모리 확인

```bash
nvidia-smi

# 확인 사항:
# - GPU 메모리 사용량이 2GB 이하인지 확인
# - 안정적으로 유지되는지 1분간 관찰
```

**안전 기준**:
- ✅ **안전**: GPU 메모리 < 2GB, 안정적
- ⚠️ **주의**: GPU 메모리 2-5GB, 증가/감소 반복
- ❌ **위험**: GPU 메모리 > 5GB, 지속적 증가

---

## 🔧 단계별 안전 실행 방법

### 방법 1: 최소 규모 테스트 (권장)

**단일 이미지만 렌더링** (GPU 메모리 ~200MB 사용)

```bash
source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter

# 단일 프레임, 단일 뷰 렌더링 (15-20초)
python3 render_image.py configs/markerless_mouse_nerf.json 0 0 \
    --out_fn output/markerless_mouse_nerf/test_single.png

# 성공하면:
echo "✓ Single render successful"
ls -lh output/markerless_mouse_nerf/test_single.png
```

### 방법 2: 소규모 배치 (메모리 안정 시)

**3개 이미지만 렌더링** (GPU 메모리 ~500MB 사용)

```bash
source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter

# 멀티뷰 3개만
for view in 0 1 2; do
    echo "Rendering view $view"
    python3 render_image.py configs/markerless_mouse_nerf.json 0 $view \
        --out_fn output/markerless_mouse_nerf/renders/multiview/frame0000_view${view}.png

    # 각 렌더링 후 1초 대기 (메모리 정리)
    sleep 1
done

echo "✓ Multi-view (3 views) complete"
ls -lh output/markerless_mouse_nerf/renders/multiview/
```

### 방법 3: 전체 파이프라인 (메모리 충분 시만)

**GPU 메모리가 1GB 이하이고 안정적일 때만 실행**

```bash
# 1. 메모리 확인
nvidia-smi | grep "MiB /"

# 2. 안전하면 실행
source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter
bash run_all_visualization.sh

# 3. 다른 터미널에서 모니터링
watch -n 5 nvidia-smi
```

---

## 🛡️ 안전 장치

### 자동 메모리 체크 스크립트

```bash
#!/bin/bash
# safe_render.sh

# GPU 메모리 확인 함수
check_gpu_memory() {
    used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "Current GPU memory: ${used_memory} MB"

    if [ "$used_memory" -gt 5000 ]; then
        echo "❌ GPU memory too high (${used_memory} MB). Aborting."
        return 1
    elif [ "$used_memory" -gt 2000 ]; then
        echo "⚠️  GPU memory moderate (${used_memory} MB). Proceed with caution."
        return 0
    else
        echo "✅ GPU memory safe (${used_memory} MB)."
        return 0
    fi
}

# 메모리 체크 후 실행
if check_gpu_memory; then
    echo "Starting safe rendering..."
    source /home/joon/miniconda3/etc/profile.d/conda.sh
    conda activate splatter

    # 단일 렌더링만 수행
    python3 render_image.py configs/markerless_mouse_nerf.json 0 0 \
        --out_fn output/markerless_mouse_nerf/safe_test.png

    echo "✓ Safe rendering complete"
else
    echo "Please free GPU memory first"
    exit 1
fi
```

**사용법**:
```bash
chmod +x safe_render.sh
./safe_render.sh
```

---

## 📋 문제 발생 시 대응

### OOM 에러 발생 시

```bash
# 즉시 중단 (Ctrl+C)

# 백그라운드 프로세스 확인
ps aux | grep python3

# 필요시 프로세스 종료 (주의!)
# pkill -f render_image.py
```

### 메모리 확보 방법

**백그라운드 학습이 완료되었다면**:
```bash
# 학습 프로세스 확인
ps aux | grep train_script.py

# 완료되었는지 로그 확인
tail -20 output/markerless_mouse_nerf/logs/step6_training.log

# 완료되었으면 종료 가능
# pkill -f train_script.py
```

---

## 🎯 권장 실행 순서

### Phase 1: 검증 (안전)
```bash
# 1. 단일 렌더링 테스트
python3 render_image.py configs/markerless_mouse_nerf.json 0 0 \
    --out_fn test1.png

# 2. 성공하면 3개 렌더링
for i in 0 1 2; do
    python3 render_image.py configs/markerless_mouse_nerf.json 0 $i \
        --out_fn test_view${i}.png
    sleep 1
done
```

### Phase 2: 소규모 배치 (주의)
```bash
# GPU 메모리 < 2GB 확인 후
# 멀티뷰 6개
bash -c 'source ~/.../conda.sh && conda activate splatter && \
    python3 generate_multiview.py'
```

### Phase 3: 전체 파이프라인 (메모리 충분 시)
```bash
# GPU 메모리 < 1GB 확인 후
bash run_all_visualization.sh
```

---

## 📊 예상 리소스 사용량

| 작업 | GPU 메모리 | 소요 시간 | 안전도 |
|------|------------|-----------|--------|
| 단일 이미지 | ~200MB | 15-20초 | ✅ 매우 안전 |
| 멀티뷰 3개 | ~500MB | 1분 | ✅ 안전 |
| 멀티뷰 6개 | ~1GB | 2분 | ⚠️ 주의 |
| 시간순서 30개 | ~3GB | 10분 | ❌ 위험 |
| 전체 파이프라인 | ~5GB | 20분 | ❌ 매우 위험 |

---

## 🔍 실시간 모니터링

### 터미널 1: 실행
```bash
source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter
python3 render_image.py configs/markerless_mouse_nerf.json 0 0 --out_fn test.png
```

### 터미널 2: 모니터링
```bash
# GPU 메모리 실시간 모니터링 (2초마다)
watch -n 2 nvidia-smi

# 또는 간략 버전
watch -n 2 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

---

## ✅ 완료된 작업 (테스트 불필요)

현재까지 이미 완료된 작업:
- ✅ 모든 시각화 스크립트 구현 완료
- ✅ torch_scatter 호환성 문제 해결
- ✅ FFmpeg 설치 완료
- ✅ 24개 샘플 이미지 생성 완료
- ✅ 문서화 완료

**추가 테스트는 GPU 메모리가 안정화된 후에 수행하는 것이 안전합니다.**

---

## 📝 안전 체크리스트

실행 전 반드시 확인:
- [ ] `nvidia-smi`로 GPU 메모리 확인
- [ ] 메모리 사용량 2GB 이하인지 확인
- [ ] 1분간 메모리 변화 관찰
- [ ] 백그라운드 프로세스 상태 확인
- [ ] 작은 규모부터 시작 (단일 이미지 → 3개 → 전체)

---

**중요**: GPU 메모리가 증가/감소를 반복하는 동안에는 **추가 렌더링을 하지 마세요**. 백그라운드 학습이 완료되고 메모리가 안정화될 때까지 기다리는 것이 가장 안전합니다.
