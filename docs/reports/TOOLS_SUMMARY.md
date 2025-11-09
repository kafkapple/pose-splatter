# 준비된 도구 및 스크립트 요약

파이프라인 실행 중 GPU/CPU 리소스 간섭 없이 미리 준비한 분석 및 실험 도구들입니다.

---

## 📊 분석 스크립트

### 1. `analyze_results.py` - 종합 결과 분석
**위치**: `/home/joon/dev/pose-splatter/analyze_results.py`

**기능**:
- 메트릭 CSV 로드 및 통계 계산
- 다중 플롯 생성 (비교, 히트맵)
- Baseline 대비 성능 개선율 계산
- 텍스트 요약 리포트 생성

**실행 예시**:
```bash
python3 analyze_results.py configs/markerless_mouse_nerf.json
python3 analyze_results.py configs/markerless_mouse_nerf.json --baseline path/to/baseline.csv
```

---

### 2. `visualize_training.py` - 훈련 과정 시각화
**위치**: `/home/joon/dev/pose-splatter/visualize_training.py`

**기능**:
- 로그 파일 파싱
- Loss/PSNR 커브 플롯
- 파이프라인 타임라인 분석

**실행 예시**:
```bash
python3 visualize_training.py \
    --log_file output/markerless_mouse_nerf/logs/step6_training.log \
    --output_dir output/markerless_mouse_nerf/analysis
```

---

### 3. `visualize_renders.py` - 렌더링 결과 시각화
**위치**: `/home/joon/dev/pose-splatter/visualize_renders.py`

**기능**:
- GT vs 예측 비교 시각화
- 프레임별 다중 뷰 그리드
- 알파 채널 시각화

**실행 예시**:
```bash
# 비교 모드
python3 visualize_renders.py \
    --mode compare \
    --gt_file output/markerless_mouse_nerf/images/images.h5 \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 100 500 1000

# 그리드 모드
python3 visualize_renders.py \
    --mode grid \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 100 200 300
```

---

### 4. `compare_configs.py` - 설정 파일 비교
**위치**: `/home/joon/dev/pose-splatter/compare_configs.py`

**기능**:
- 여러 config JSON 파일 비교
- 차이점만 표시
- Markdown/LaTeX 테이블 출력

**실행 예시**:
```bash
python3 compare_configs.py \
    configs/markerless_mouse_nerf.json \
    configs/markerless_mouse_nerf_high_res.json \
    configs/markerless_mouse_nerf_fast.json \
    --format markdown
```

---

## 🔧 실험 Config 변형본

### 1. Baseline (현재 실행 중)
**파일**: `configs/markerless_mouse_nerf.json`
- image_downsample: 4x
- grid_size: 112
- lr: 1e-4
- ssim_lambda: 0.0

### 2. High Resolution
**파일**: `configs/markerless_mouse_nerf_high_res.json`
- image_downsample: 2x ⬆️
- grid_size: 128 ⬆️
- **목적**: 더 높은 이미지 품질

### 3. Fast Variant
**파일**: `configs/markerless_mouse_nerf_fast.json`
- image_downsample: 8x ⬇️
- grid_size: 64 ⬇️
- frame_jump: 10 ⬆️
- lr: 2e-4 ⬆️
- **목적**: 빠른 프로토타이핑

### 4. SSIM Loss
**파일**: `configs/markerless_mouse_nerf_ssim.json`
- img_lambda: 0.3
- ssim_lambda: 0.2 (새로 추가)
- **목적**: 구조적 유사성 개선

---

## 📁 문서화

### 1. 실험 보고서
**파일**: `docs/reports/251109_experiment_baseline.md`

**내용**:
- 실험 개요 및 목적
- 상세 설정 정보
- 파이프라인 각 단계 상태
- 예상 출력 파일
- 모니터링 방법
- 체크리스트

### 2. 분석 가이드
**파일**: `docs/reports/ANALYSIS_GUIDE.md`

**내용**:
- 각 분석 도구 사용법
- 실험 결과 분석 워크플로우
- 메트릭 해석 가이드
- 문제 진단 팁
- 권장 폴더 구조

---

## 🚀 다음 단계 실행 방법

### 파이프라인 완료 후

1. **결과 분석**
```bash
cd /home/joon/dev/pose-splatter
python3 analyze_results.py configs/markerless_mouse_nerf.json
```

2. **시각화 생성**
```bash
python3 visualize_training.py \
    --log_file output/markerless_mouse_nerf/logs/step6_training.log \
    --output_dir output/markerless_mouse_nerf/analysis

python3 visualize_renders.py \
    --mode compare \
    --gt_file output/markerless_mouse_nerf/images/images.h5 \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 500 1000 1500 2000 \
    --output_dir output/markerless_mouse_nerf/visualization
```

3. **다음 실험 실행**
```bash
# High resolution 실험
bash run_pipeline_auto.sh configs/markerless_mouse_nerf_high_res.json

# Fast variant 실험
bash run_pipeline_auto.sh configs/markerless_mouse_nerf_fast.json

# SSIM loss 실험
bash run_pipeline_auto.sh configs/markerless_mouse_nerf_ssim.json
```

4. **결과 비교**
```bash
python3 compare_configs.py \
    configs/markerless_mouse_nerf.json \
    configs/markerless_mouse_nerf_high_res.json \
    --format markdown \
    --output docs/reports/config_comparison.md

python3 analyze_results.py configs/markerless_mouse_nerf_high_res.json \
    --baseline output/markerless_mouse_nerf/metrics_test.csv
```

---

## 📦 필요한 추가 패키지

분석 스크립트 실행을 위해:
```bash
pip install matplotlib seaborn pandas tabulate
```

모든 패키지가 이미 설치되어 있는지 확인:
```bash
python3 -c "import matplotlib, seaborn, pandas; print('All packages OK')"
```

---

## ⚙️ 현재 파이프라인 상태

**실행 시작**: 2025-11-09 13:54:29

**진행 상황**:
- ✅ Step 2: Center & Rotation (완료, 9분 14초 소요)
- ✅ Step 3: Crop Indices (완료, 1초 소요)
- 🔄 Step 4: Write Images to HDF5 (진행 중, 예상 2-4시간)
- ⏳ Step 5: Zarr 변환
- ⏳ Step 6: 모델 훈련 (50 epochs)
- ⏳ Step 7: 평가
- ⏳ Step 8: 렌더링

**모니터링**:
```bash
# 실시간 모니터링
watch -n 10 ./monitor_pipeline.sh

# 특정 로그 확인
tail -f output/markerless_mouse_nerf/logs/step4_write_images.log
```

---

**작성일**: 2025-11-09
**작성자**: Claude Code
