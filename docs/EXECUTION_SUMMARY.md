# Pose Splatter 전체 파이프라인 실행 요약

**실행 시작**: 2025-11-09 00:45 KST
**예상 완료**: 2025-11-09 10:00~14:00 KST (약 10-14시간 후)

---

## 📊 실행 현황

### ✅ 완료된 작업

1. **환경 설정**
   - 카메라 파라미터 변환 (pickle → HDF5)
   - Config 파일 생성 및 최적화
   - Up direction 자동 계산

2. **스크립트 준비**
   - `run_pipeline_auto.sh`: 전체 자동 실행 스크립트
   - `monitor_pipeline.sh`: 실시간 모니터링 스크립트
   - 모든 로그 파일 자동 저장

### ⏳ 현재 진행 중

**Step 2/7: Calculate Center & Rotation**
- 시작 시간: 00:45 KST
- 처리 프레임: 18,000개 (frame_jump=5) → 약 3,600개 처리
- 예상 소요: 1-2시간
- GPU 사용률: 28% (정상)

---

## 📋 전체 파이프라인 단계

| 단계 | 작업 | 예상 시간 | 상태 |
|------|------|-----------|------|
| Step 1 | Up direction 계산 | - | ✅ 완료 |
| **Step 2** | **Center & Rotation** | **1-2시간** | ⏳ **진행 중** |
| Step 3 | Volume crop indices | 1-2시간 | ⏸️ 대기 |
| Step 4 | 이미지 HDF5 저장 | 2-4시간 | ⏸️ 대기 |
| Step 5 | Zarr 변환 | 30-60분 | ⏸️ 대기 |
| Step 6 | 모델 학습 (50 epochs) | 4-8시간 | ⏸️ 대기 |
| Step 7 | 모델 평가 | 30-60분 | ⏸️ 대기 |
| Step 8 | 샘플 렌더링 | 10-20분 | ⏸️ 대기 |

**총 예상 시간**: 약 10-14시간

---

## 🖥️ 시스템 정보

- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **GPU 온도**: 65°C
- **GPU 사용률**: 28% (메모리 10.2GB/12GB)
- **프로세스 ID**: 444173

---

## 📁 데이터 정보

### 입력 데이터
```
data/markerless_mouse_1_nerf/
├── videos_undist/           # 6개 카메라 RGB 비디오
│   ├── 0.mp4 (25MB)
│   ├── 1.mp4 (17MB)
│   ├── 2.mp4 (23MB)
│   ├── 3.mp4 (21MB)
│   ├── 4.mp4 (19MB)
│   └── 5.mp4 (24MB)
├── simpleclick_undist/      # 6개 실루엣 마스크
│   ├── 0.mp4 (11MB) ~ 5.mp4
├── camera_params.h5         # 카메라 캘리브레이션
└── vertical_lines.npz       # Up direction
```

**비디오 스펙**:
- 해상도: 1152 × 1024
- FPS: 100
- 총 프레임: 18,000
- **처리 프레임**: 3,600 (frame_jump=5)

### 출력 데이터 (생성 예정)
```
output/markerless_mouse_nerf/
├── images/
│   ├── images.h5            # HDF5 이미지 (예상 크기: ~10-20GB)
│   └── images.zarr/         # Zarr 이미지 (학습용)
├── center_rotation.npz      # 각 프레임의 center & angle
├── volume_sum.npy           # Volume 통계
├── checkpoint.pt            # 학습된 모델 (예상: ~500MB)
├── metrics_test.csv         # 평가 메트릭 (IoU, SSIM, PSNR, L1)
├── renders/                 # 렌더링 결과 이미지
│   ├── render_100_*.png
│   ├── render_500_*.png
│   └── render_1000_*.png
└── logs/                    # 모든 로그 파일
    ├── pipeline_master.log
    ├── step2_center_rotation.log
    ├── step3_crop_indices.log
    ├── step4_write_images.log
    ├── step5_zarr.log
    ├── step6_training.log
    ├── step7_evaluation.log
    └── step8_rendering.log
```

---

## 📊 모니터링 방법

### 실시간 모니터링
```bash
# 10초마다 자동 업데이트
watch -n 10 ./monitor_pipeline.sh

# 마스터 로그 실시간 보기
tail -f output/markerless_mouse_nerf/logs/pipeline_master.log

# 특정 단계 로그 보기
tail -f output/markerless_mouse_nerf/logs/step2_center_rotation.log
```

### 수동 확인
```bash
# 전체 상태 확인
./monitor_pipeline.sh

# 프로세스 확인
ps aux | grep python3

# GPU 확인
nvidia-smi

# 디스크 사용량
du -sh output/markerless_mouse_nerf/*
```

### PID 파일
```bash
# 파이프라인 PID 확인
cat output/markerless_mouse_nerf/pipeline.pid

# 프로세스 종료 (필요시)
kill $(cat output/markerless_mouse_nerf/pipeline.pid)
```

---

## ⚙️ Config 설정

**파일**: `configs/markerless_mouse_nerf.json`

주요 파라미터:
```json
{
  "frame_jump": 5,              // 5프레임마다 1개 샘플링 (3,600개)
  "image_downsample": 4,        // 해상도 1/4 (288×256)
  "grid_size": 112,             // Voxel resolution
  "ell": 0.22,                  // Volume 크기 (m)
  "holdout_views": [5, 1],      // 테스트용 카메라 (novel view)
  "lr": 1e-4,                   // Learning rate
  "img_lambda": 0.5,            // Image loss weight
  "ssim_lambda": 0.0            // SSIM loss (비활성화)
}
```

---

## 🔍 예상 결과

### 정량적 메트릭 (목표)
- **IoU** (Silhouette): > 0.90
- **SSIM** (구조 유사도): > 0.85
- **PSNR** (픽셀 품질): > 25 dB
- **L1** (픽셀 오차): < 0.05

### 시각적 품질
- Photorealistic 렌더링
- Novel view synthesis (holdout cameras 5, 1)
- 다양한 자세 변형 가능

---

## 🚨 문제 해결

### 파이프라인 중단 시
```bash
# 로그 확인
tail -100 output/markerless_mouse_nerf/logs/pipeline_master.log

# 특정 단계부터 재개 (예: Step 4부터)
python3 write_images.py configs/markerless_mouse_nerf.json
python3 copy_to_zarr.py output/markerless_mouse_nerf/images/images.h5 \
                        output/markerless_mouse_nerf/images/images.zarr
python3 train_script.py configs/markerless_mouse_nerf.json --epochs 50
```

### GPU 메모리 부족 시
```json
// config.json 수정
"image_downsample": 8,    // 4 → 8
"grid_size": 64,          // 112 → 64
```

### 디스크 공간 부족 시
```bash
# 임시 파일 삭제
rm output/markerless_mouse_nerf/logs/*.log.old

# 압축 레벨 조정
"image_compression_level": 4,  // 2 → 4 (더 압축)
```

---

## 📈 진행 체크포인트

파이프라인 실행 중 다음 시점에 확인:

### 1시간 후 (01:45 KST)
- [ ] Step 2 완료 확인
- [ ] `center_rotation.npz` 파일 생성 확인
- [ ] Step 3 시작 확인

### 3시간 후 (03:45 KST)
- [ ] Step 3 완료 확인
- [ ] `volume_idx` 자동 업데이트 확인
- [ ] Step 4 시작 확인 (가장 긴 단계)

### 6시간 후 (06:45 KST)
- [ ] Step 4-5 완료 확인
- [ ] `images.h5`, `images.zarr` 생성 확인
- [ ] Step 6 (학습) 시작 확인
- [ ] GPU 사용률 > 80% 확인

### 10시간 후 (10:45 KST)
- [ ] Step 6 완료 확인
- [ ] `checkpoint.pt` 생성 확인
- [ ] 학습 loss 곡선 확인 (`loss.pdf`)
- [ ] Step 7-8 완료 확인

### 최종 확인 (완료 시)
- [ ] `metrics_test.csv` 확인
- [ ] 렌더링 이미지 품질 확인
- [ ] 모든 로그 파일 정상 종료 확인

---

## 📝 다음 단계 (파이프라인 완료 후)

1. **결과 분석**
   ```bash
   # 메트릭 확인
   cat output/markerless_mouse_nerf/metrics_test.csv

   # 렌더링 이미지 확인
   ls -lh output/markerless_mouse_nerf/renders/
   ```

2. **추가 렌더링**
   ```bash
   # 다양한 자세로 렌더링
   python3 render_image.py configs/markerless_mouse_nerf.json 100 0 --angle_offset 0.5
   python3 render_image.py configs/markerless_mouse_nerf.json 100 0 --delta_x 0.1
   ```

3. **시각 특징 추출**
   ```bash
   python3 calculate_visual_features.py configs/markerless_mouse_nerf.json
   python3 calculate_visual_embedding.py configs/markerless_mouse_nerf.json
   ```

4. **결과 문서화**
   - 연구 노트 작성
   - 메트릭 비교 분석
   - 시각적 결과 정리

---

## 📞 연락처 및 참고

- **프로젝트 디렉토리**: `/home/joon/dev/pose-splatter`
- **데이터 디렉토리**: `/home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf`
- **출력 디렉토리**: `/home/joon/dev/pose-splatter/output/markerless_mouse_nerf`
- **관련 문서**:
  - `ANALYSIS_REPORT.md`: 기술 분석 보고서
  - `SETUP_GUIDE.md`: 환경 설정 가이드
  - `QUICKSTART.md`: 빠른 시작 가이드

---

**작성**: 2025-11-09 00:47 KST
**상태**: 파이프라인 실행 중 ⏳
**예상 완료**: 2025-11-09 10:00~14:00 KST
