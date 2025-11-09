# Pose Splatter 실행 요약

**날짜**: 2025-11-09
**상태**: 훈련 진행 중 ✅

---

## 📋 실행 결과

### 파이프라인 진행 상황

| 단계 | 작업 | 상태 | 소요 시간 | 비고 |
|------|------|------|-----------|------|
| Step 1 | Up direction | ⏭️ | - | 스킵 (사전 계산됨) |
| Step 2 | Center & Rotation | ✅ | 9분 14초 | 18,000 프레임 처리 |
| Step 3 | Crop Indices | ✅ | 1초 | volume_idx 계산 |
| Step 4 | Write Images (HDF5) | ✅ | 5분 55초 | 195MB 생성 |
| Step 5 | Convert to Zarr | ✅ | 약 1분 | 수동 실행 |
| Step 6 | Model Training | 🔄 | 진행 중 | 50 epochs (예상 4-8시간) |
| Step 7 | Evaluation | ⏳ | 대기 중 | - |
| Step 8 | Rendering | ⏳ | 대기 중 | - |

---

## 🔧 해결한 문제들

### 1. GPU 미사용 이슈 ✅
**문제**: Step 2-4에서 GPU 사용률 0%
**해결**: 정상 동작 확인
- Step 2-4는 CPU 전용 작업 (영상 처리, shape carving)
- Step 6 (훈련)부터 GPU 본격 사용 (현재 1.2GB VRAM 사용 확인)

### 2. 누락 패키지 설치 ✅
다음 패키지들이 누락되어 수동 설치:
```bash
pip install gsplat                    # Gaussian Splatting 핵심 라이브러리
pip install torch-scatter             # Scatter 연산용
# torchmetrics, zarr, h5py 등은 이미 설치됨
```

### 3. Zarr 변환 에러 ✅
**문제**: `ContainsGroupError: path '' contains a group`
**원인**: 기존 zarr 파일 존재
**해결**:
```bash
rm -rf output/markerless_mouse_nerf/images/images.zarr
python3 copy_to_zarr.py [input] [output]
```

---

## 💻 현재 시스템 상태

### GPU 사용 현황
```
GPU: NVIDIA GeForce RTX 3060 (12GB)
메모리 사용: 1520 MiB / 12288 MiB (12.4%)
GPU 이용률: 훈련 초기화 중
온도: 47°C
전력: 17W / 170W
```

### 훈련 설정
```
Config: configs/markerless_mouse_nerf.json
Epochs: 50
Workers: 12
Batch size: 기본값
Learning rate: 1e-4
```

---

## 📁 생성된 파일

```
output/markerless_mouse_nerf/
├── center_rotation.npz          (367KB) - Center & rotation 데이터
├── vertical_lines.npz           (282B)  - Up direction
├── images/
│   ├── images.h5                (195MB) - HDF5 형식 이미지
│   └── images.zarr/             - Zarr 형식 (훈련용)
└── logs/
    ├── step2_center_rotation.log
    ├── step3_crop_indices.log
    ├── step4_write_images.log
    ├── step5_zarr.log
    └── step6_training.log       - 현재 진행 중
```

---

## 📊 모니터링 방법

### 실시간 훈련 모니터링
```bash
# 훈련 로그 확인
tail -f output/markerless_mouse_nerf/logs/step6_training.log

# GPU 상태 (2초마다 갱신)
watch -n 2 nvidia-smi

# 파이프라인 전체 상태 (10초마다 갱신)
watch -n 10 ./monitor_pipeline.sh
```

### 예상 완료 시간
- **Step 6 (훈련)**: 4-8시간 (50 epochs)
- **Step 7 (평가)**: ~10분
- **Step 8 (렌더링)**: ~5분

**총 예상 완료**: 약 5-9시간 후

---

## 🎯 다음 단계

### 훈련 완료 후
1. **결과 분석**
```bash
python3 analyze_results.py configs/markerless_mouse_nerf.json
```

2. **시각화**
```bash
python3 visualize_training.py \
    --log_file output/markerless_mouse_nerf/logs/step6_training.log

python3 visualize_renders.py \
    --mode compare \
    --gt_file output/markerless_mouse_nerf/images/images.h5 \
    --pred_file output/markerless_mouse_nerf/images/rendered_images.h5 \
    --frames 0 500 1000
```

3. **추가 실험 (선택사항)**
```bash
# High resolution 실험
bash run_pipeline_auto.sh configs/markerless_mouse_nerf_high_res.json

# Fast variant
bash run_pipeline_auto.sh configs/markerless_mouse_nerf_fast.json

# SSIM loss
bash run_pipeline_auto.sh configs/markerless_mouse_nerf_ssim.json
```

---

## 📝 학습 내용

### 파이프라인 구조
1. **전처리 단계** (Step 2-5): CPU 집약적
   - 비디오 프레임 읽기 및 처리
   - Shape carving을 통한 3D 볼륨 생성
   - HDF5/Zarr 데이터 저장

2. **훈련 단계** (Step 6): GPU 집약적
   - 3D Gaussian Splatting 모델 훈련
   - 실시간 렌더링 및 손실 계산
   - 체크포인트 저장

3. **평가 단계** (Step 7-8): GPU 사용
   - 테스트 세트 렌더링
   - 메트릭 계산 (PSNR, SSIM, IoU)
   - 샘플 이미지 생성

### 중요 패키지
- **gsplat**: Gaussian Splatting 렌더링
- **torch-scatter**: Scatter 연산 (PyG)
- **torchmetrics**: SSIM 등 메트릭 계산
- **zarr**: 고속 배열 I/O

---

## ✅ 체크리스트

- [x] 환경 및 데이터 검증
- [x] 누락 패키지 설치
- [x] Step 2: Center & Rotation 계산
- [x] Step 3: Crop Indices 계산
- [x] Step 4: HDF5 이미지 저장
- [x] Step 5: Zarr 변환
- [x] Step 6: 모델 훈련 시작
- [ ] Step 6: 모델 훈련 완료 (진행 중)
- [ ] Step 7: 모델 평가
- [ ] Step 8: 샘플 렌더링
- [ ] 결과 분석 및 시각화
- [ ] 실험 문서화 완료

---

## 📚 생성된 도구 및 문서

### 분석 스크립트 (4개)
- `analyze_results.py` - 종합 결과 분석
- `visualize_training.py` - 훈련 과정 시각화
- `visualize_renders.py` - 렌더링 결과 비교
- `compare_configs.py` - 설정 파일 비교

### Config 변형 (3개)
- `markerless_mouse_nerf_high_res.json` - 고해상도
- `markerless_mouse_nerf_fast.json` - 빠른 실험
- `markerless_mouse_nerf_ssim.json` - SSIM 손실

### 문서 (4개)
- `README.md` - 업데이트된 사용 가이드
- `docs/reports/251109_experiment_baseline.md` - 실험 보고서
- `docs/reports/ANALYSIS_GUIDE.md` - 분석 가이드
- `docs/reports/TOOLS_SUMMARY.md` - 도구 요약

---

**작성 시간**: 2025-11-09 14:13 KST
**마지막 업데이트**: 훈련 시작 확인
