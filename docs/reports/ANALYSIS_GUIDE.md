# Analysis & Visualization Guide

실험 완료 후 결과 분석 및 시각화를 위한 가이드입니다.

---

## 🔧 준비된 분석 도구

### 1. 결과 종합 분석 (`analyze_results.py`)

**기능**:
- 메트릭 CSV 파일 로드 및 분석
- 다양한 시각화 (비교 플롯, 히트맵)
- 통계 요약 리포트 생성
- Baseline 비교

**사용법**:
```bash
# 기본 분석
python3 analyze_results.py configs/markerless_mouse_nerf.json

# Baseline과 비교
python3 analyze_results.py configs/markerless_mouse_nerf.json \
    --baseline output/baseline/metrics_test.csv \
    --output_dir output/markerless_mouse_nerf/analysis

# 출력 결과
output/markerless_mouse_nerf/analysis/
├── metrics_comparison.png    # 메트릭 비교 플롯
├── metrics_heatmap.png        # 히트맵
├── test/                      # 테스트 분할 상세
└── analysis_summary.txt       # 텍스트 요약
```

**생성되는 메트릭**:
- L1 Loss (낮을수록 좋음)
- IoU (높을수록 좋음)
- Soft IoU (높을수록 좋음)
- PSNR (높을수록 좋음)
- SSIM (높을수록 좋음)

---

### 2. 훈련 과정 시각화 (`visualize_training.py`)

**기능**:
- 훈련 로그 파싱
- Loss/PSNR 커브 플롯
- 파이프라인 타임라인 시각화

**사용법**:
```bash
# 훈련 커브 생성
python3 visualize_training.py \
    --log_file output/markerless_mouse_nerf/logs/step6_training.log \
    --output_dir output/markerless_mouse_nerf/analysis

# 출력 결과
output/markerless_mouse_nerf/analysis/
└── training_curves.png  # Loss 및 PSNR 커브
```

---

### 3. 렌더링 결과 시각화 (`visualize_renders.py`)

**기능**:
- Ground truth vs 예측 비교
- 프레임별 그리드 시각화
- 알파 채널 시각화

**사용법**:
```bash
# GT vs 예측 비교
python3 visualize_renders.py \
    --mode compare \
    --gt_file output/markerless_mouse_nerf/images/images.h5 \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 100 500 1000 \
    --output_dir output/markerless_mouse_nerf/visualization

# 프레임 그리드만 생성
python3 visualize_renders.py \
    --mode grid \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 100 200 300 400 500 \
    --output_dir output/markerless_mouse_nerf/visualization

# 출력 결과
output/markerless_mouse_nerf/visualization/
├── comparison_frame_00000.png
├── comparison_frame_00100.png
├── comparison_frame_00500.png
├── comparison_frame_01000.png
├── pred_grids/
│   ├── frame_00000.png
│   └── ...
└── gt_grids/
    └── ...
```

---

### 4. 설정 파일 비교 (`compare_configs.py`)

**기능**:
- 여러 config 파일 비교
- 차이점 하이라이트
- 표 형식 출력

**사용법**:
```bash
# 여러 설정 비교
python3 compare_configs.py \
    configs/markerless_mouse_nerf.json \
    configs/markerless_mouse_nerf_high_res.json \
    configs/markerless_mouse_nerf_fast.json \
    --format markdown \
    --output config_comparison.md

# 출력: 차이가 있는 파라미터만 표시
```

---

## 📊 실험 결과 분석 워크플로우

### Step 1: 기본 메트릭 분석
```bash
cd /home/joon/dev/pose-splatter

# 종합 분석 실행
python3 analyze_results.py configs/markerless_mouse_nerf.json \
    --output_dir output/markerless_mouse_nerf/analysis
```

### Step 2: 훈련 과정 검토
```bash
# 훈련 커브 시각화
python3 visualize_training.py \
    --log_file output/markerless_mouse_nerf/logs/step6_training.log \
    --output_dir output/markerless_mouse_nerf/analysis
```

### Step 3: 시각적 품질 평가
```bash
# 샘플 프레임 비교
python3 visualize_renders.py \
    --mode compare \
    --gt_file output/markerless_mouse_nerf/images/images.h5 \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 500 1000 1500 2000 \
    --output_dir output/markerless_mouse_nerf/visualization
```

### Step 4: 분석 리포트 확인
```bash
# 요약 텍스트 파일 확인
cat output/markerless_mouse_nerf/analysis/analysis_summary.txt

# 시각화 결과 확인 (이미지 뷰어에서)
# - metrics_comparison.png
# - training_curves.png
# - comparison_frame_*.png
```

---

## 🔍 주요 체크포인트

### 메트릭 해석
1. **PSNR > 25dB**: 일반적으로 좋은 품질
2. **SSIM > 0.8**: 구조적 유사성 높음
3. **IoU > 0.7**: 마스크 정확도 양호
4. **L1 < 0.1**: 픽셀 단위 오차 작음

### 문제 진단
- **훈련 Loss가 떨어지지 않음**: Learning rate 조정 필요
- **Validation Loss 증가**: Overfitting, regularization 필요
- **특정 뷰에서만 낮은 성능**: 카메라 캘리브레이션 확인
- **알파 채널 문제**: volume_fill_color 조정

---

## 📁 권장 폴더 구조

```
output/markerless_mouse_nerf/
├── checkpoint.pt              # 모델 체크포인트
├── metrics_test.csv           # 테스트 메트릭
├── metrics_train.csv          # 훈련 메트릭
├── metrics_valid.csv          # 검증 메트릭
├── logs/                      # 모든 로그 파일
├── images/
│   ├── images.h5              # 원본 이미지
│   ├── images.zarr            # Zarr 형식
│   └── rendered_images.h5     # 렌더링 결과
├── analysis/                  # 분석 결과
│   ├── metrics_comparison.png
│   ├── metrics_heatmap.png
│   ├── training_curves.png
│   └── analysis_summary.txt
└── visualization/             # 시각화 결과
    ├── comparison_*.png
    ├── pred_grids/
    └── gt_grids/
```

---

## 💡 유용한 팁

### 빠른 메트릭 확인
```bash
# CSV를 테이블로 보기
column -s, -t < output/markerless_mouse_nerf/metrics_test.csv | less -S
```

### GPU 메모리 사용량 확인
```bash
# 평가 중 GPU 모니터링
watch -n 1 nvidia-smi
```

### 로그에서 에러 찾기
```bash
# 모든 로그에서 에러 검색
grep -i error output/markerless_mouse_nerf/logs/*.log
```

---

## 📦 필수 패키지

분석 스크립트 실행을 위해 필요한 패키지:

```bash
pip install matplotlib seaborn pandas tabulate h5py
```

---

**Last Updated**: 2025-11-09
