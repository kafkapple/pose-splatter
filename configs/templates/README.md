# Config Templates

이 디렉토리는 다양한 환경과 용도에 맞는 config 템플릿을 제공합니다.

## 사용 방법

### 1. 템플릿 복사

```bash
# 원하는 템플릿 복사
cp configs/templates/rtx3060_3d.json configs/experiments/my_experiment.json

# 필요시 수정
vim configs/experiments/my_experiment.json
```

### 2. 직접 실행

```bash
# 템플릿 직접 사용
bash scripts/training/run_training.sh configs/templates/rtx3060_3d.json --epochs 50
```

---

## 템플릿 목록

### **rtx3060_3d.json** - RTX 3060 12GB (3D GS)

**대상 GPU**: RTX 3060, RTX 3070, RTX 2080 Ti
**메모리**: 12GB
**특징**: 메모리 효율적, 안정적

```json
{
  "image_downsample": 4,      // 288×256 pixels
  "grid_size": 112,
  "gaussian_mode": "3d"
}
```

**예상 성능**:
- 학습 시간: 6-8 시간 (50 epochs)
- GPU 사용량: ~6-8GB
- PSNR: ~25-27 dB

**사용 예**:
```bash
bash scripts/training/run_training.sh configs/templates/rtx3060_3d.json --epochs 50
```

---

### **a6000_2d.json** - A6000 24GB+ (2D GS High Quality)

**대상 GPU**: A6000, RTX 4090, RTX 3090
**메모리**: 24GB+
**특징**: 고품질, SSIM loss 사용

```json
{
  "image_downsample": 2,      // 576×512 pixels
  "grid_size": 128,
  "gaussian_mode": "2d",
  "gaussian_config": {
    "sigma_cutoff": 3.0,
    "kernel_size": 5,
    "batch_size": 5
  },
  "ssim_lambda": 0.1
}
```

**예상 성능**:
- 학습 시간: 10-15 시간 (50 epochs)
- GPU 사용량: ~18-20GB
- PSNR: ~28-30 dB

**사용 예**:
```bash
bash scripts/training/run_training.sh configs/templates/a6000_2d.json --epochs 50
```

---

### **debug_quick.json** - 빠른 디버그 (모든 GPU)

**대상 GPU**: 모든 GPU
**메모리**: 4GB+
**특징**: 빠른 테스트, 100프레임만 사용

```json
{
  "image_downsample": 6,      // 192×171 pixels
  "grid_size": 64,
  "max_frames": 100,
  "frame_jump": 10
}
```

**예상 성능**:
- 학습 시간: 5-10 분 (5 epochs)
- GPU 사용량: ~2-3GB

**사용 예**:
```bash
# 5 epochs만 빠르게 테스트
bash scripts/training/run_training.sh configs/templates/debug_quick.json --epochs 5
```

---

## 주요 파라미터 설명

### Gaussian Mode

- `"gaussian_mode": "3d"` - 3D GS (gsplat), 메모리 효율적
- `"gaussian_mode": "2d"` - 2D GS (custom), 고품질

### Image Resolution

| downsample | 해상도 | GPU 메모리 | 품질 |
|------------|--------|-----------|------|
| 2 | 576×512 | 높음 | 최고 |
| 4 | 288×256 | 보통 | 좋음 |
| 6 | 192×171 | 낮음 | 괜찮음 |

### Grid Size

| grid_size | 복셀 수 | 메모리 | 품질 |
|-----------|---------|--------|------|
| 64 | 262K | 낮음 | 낮음 |
| 112 | 1.4M | 보통 | 좋음 |
| 128 | 2.1M | 높음 | 매우 좋음 |

---

## 커스텀 Config 만들기

### Python 스크립트로 생성

```python
import json

# 베이스 템플릿 로드
with open('configs/templates/rtx3060_3d.json') as f:
    config = json.load(f)

# 수정
config['project_directory'] = 'output/my_custom_experiment/'
config['lr'] = 5e-5
config['ssim_lambda'] = 0.1

# 저장
with open('configs/experiments/my_custom.json', 'w') as f:
    json.dump(config, f, indent=4)
```

### 명령줄에서 간단히 수정

```bash
# project_directory만 변경
cat configs/templates/rtx3060_3d.json | \
  sed 's|output/rtx3060_3d/|output/my_experiment/|' > \
  configs/experiments/my_experiment.json
```

---

## 비교표

| 템플릿 | GPU | Mode | downsample | grid | 시간 | PSNR |
|--------|-----|------|------------|------|------|------|
| rtx3060_3d | 12GB | 3D | 4 | 112 | 6-8h | 25-27 |
| a6000_2d | 24GB | 2D | 2 | 128 | 10-15h | 28-30 |
| debug_quick | 4GB | 3D | 6 | 64 | 5-10m | N/A |

---

## 문제 해결

### CUDA Out of Memory

**증상**: `RuntimeError: CUDA out of memory`

**해결**:
```bash
# downsample 증가 (해상도 감소)
"image_downsample": 6  # 4 → 6

# grid_size 감소
"grid_size": 96  # 112 → 96

# 2D의 경우 batch_size 감소
"gaussian_config": {
  "batch_size": 3  # 5 → 3
}
```

### 학습이 너무 느림

**해결**:
```bash
# frame_jump 증가 (프레임 수 감소)
"frame_jump": 10  # 5 → 10

# max_frames 설정
"max_frames": 1000

# validation 빈도 감소
"valid_every": 10  # 5 → 10
```

### 품질이 낮음

**해결**:
```bash
# downsample 감소 (해상도 증가)
"image_downsample": 2  # 4 → 2

# grid_size 증가
"grid_size": 128  # 112 → 128

# SSIM loss 추가
"ssim_lambda": 0.1  # 0.0 → 0.1
```
