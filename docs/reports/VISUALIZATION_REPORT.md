# Pose Splatter 시각화 구현 보고서

**날짜**: 2025-11-10
**작업자**: Claude Code
**프로젝트**: Pose Splatter - 3D Gaussian Splatting 시각화

---

## 1. 목적

학습된 Pose Splatter 모델을 기반으로 다양한 관점에서 결과를 시각화하여 3D 재구성 품질을 검증하고, 연구 결과를 효과적으로 제시하기 위한 시각화 파이프라인 구축

---

## 2. 구현 내용

### 2.1 시각화 유형

4가지 주요 시각화 방식을 구현:

#### (1) 멀티뷰 렌더링 (Multi-view Rendering)
- **목적**: 동일 프레임을 6개의 서로 다른 카메라 각도에서 렌더링
- **구현 파일**: `generate_multiview.py`
- **출력**: `output/markerless_mouse_nerf/renders/multiview/`
- **예상 결과**: frame0000_view0.png ~ frame0000_view5.png (6개 이미지)

#### (2) 시간순서 시퀀스 (Temporal Sequence)
- **목적**: 연속된 프레임을 렌더링하여 동작 재현
- **구현 파일**: `generate_temporal_video.py`
- **출력**: `output/markerless_mouse_nerf/renders/temporal/`
- **예상 결과**: frame0000.png ~ frame0029.png (30개 이미지) + temporal_sequence.mp4 (비디오)

#### (3) 360도 회전 뷰 (360-degree Rotation)
- **목적**: 단일 프레임을 중심으로 카메라를 360도 회전시키며 렌더링
- **구현 파일**: `generate_360_rotation.py`
- **출력**: `output/markerless_mouse_nerf/renders/rotation360/`
- **예상 결과**: rot000.png ~ rot023.png (24개 이미지) + rotation360.mp4 (비디오)

#### (4) 3D 포인트 클라우드 (Point Cloud Export)
- **목적**: 3D Gaussian Splatting 결과를 PLY 형식으로 내보내기
- **구현 파일**: `export_point_cloud.py`
- **출력**: `output/markerless_mouse_nerf/pointclouds/frame0000.ply`
- **내용**: 3D 위치, RGB 색상, 투명도(opacity) 정보 포함

### 2.2 통합 실행 스크립트

**파일**: `run_all_visualization.sh`

모든 시각화 작업을 순차적으로 실행하는 bash 스크립트:
```bash
#!/bin/bash
source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter

# 1. Multi-view rendering
# 2. Temporal sequence + video creation (ffmpeg)
# 3. 360-degree rotation + video creation (ffmpeg)
# 4. Point cloud export
```

---

## 3. 기술적 문제 해결

### 3.1 torch_scatter 라이브러리 호환성 문제

**증상**:
```
OSError: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
```

**원인**:
- 시스템 전역 torch_scatter (PyTorch 2.0 + CUDA 11.8)와 conda 환경의 torch_scatter (PyTorch 2.6 + CUDA 12.4) 버전 불일치

**해결책**:
```bash
# 기존 버전 제거
pip uninstall torch_scatter -y

# PyTorch 2.6.0 + CUDA 12.4 호환 버전 설치
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

**결과**: torch-scatter 2.1.2+pt26cu124 설치 완료

### 3.2 GPU 메모리 부족 (CUDA OOM)

**증상**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 MiB.
GPU 0 has a total capacity of 11.75 GiB of which 19.75 MiB is free.
```

**원인**:
- 백그라운드 학습/평가 프로세스들이 GPU 메모리 ~11GB 점유
- 렌더링 작업 수행 시 추가 메모리 필요

**영향**:
- 멀티뷰: 6개 중 3개만 성공 (view 0, 1, 2)
- 시간순서: 30개 중 3개만 성공 (frame 5, 6, 7)
- 360도 회전: 24개 중 18개 성공 (rot003 ~ rot020)
- 포인트 클라우드: 실패

**해결 방안**:
1. 백그라운드 프로세스 종료 후 재실행
2. 배치 크기 축소 또는 순차 실행
3. GPU 메모리 정리: `torch.cuda.empty_cache()` 추가

### 3.3 FFmpeg libx264 인코더 부재

**증상**:
```
[vost#0:0 @ 0xfd4bb80] Unknown encoder 'libx264'
Error opening output files: Encoder not found
```

**원인**: conda 환경에 x264 코덱이 설치되지 않음

**해결책**:
```bash
conda install -c conda-forge x264 ffmpeg
```

**현재 상태**: 미설치 (비디오 생성 불가)

---

## 4. 실행 결과

### 4.1 성공적으로 생성된 파일

#### 멀티뷰 렌더링 (3/6 완료)
```
output/markerless_mouse_nerf/renders/multiview/
├── frame0000_view0.png (76KB)
├── frame0000_view1.png (87KB)
└── frame0000_view2.png (79KB)
```

#### 시간순서 시퀀스 (3/30 완료)
```
output/markerless_mouse_nerf/renders/temporal/
├── frame0005.png (81KB)
├── frame0006.png (81KB)
└── frame0007.png (83KB)
```

#### 360도 회전 (18/24 완료)
```
output/markerless_mouse_nerf/renders/rotation360/
├── rot003.png (88KB)
├── rot004.png (85KB)
├── ...
└── rot020.png (65KB)
```

**총 24개 이미지 생성 완료** (1.9MB)

### 4.2 생성 실패 항목

- 멀티뷰 view 3, 4, 5 (GPU OOM)
- 시간순서 frame 0-4, 8-29 (GPU OOM)
- 360도 회전 rot000-002, rot021-023 (GPU OOM)
- 포인트 클라우드 전체 (GPU OOM)
- 비디오 파일 (FFmpeg 인코더 부재)

---

## 5. 렌더링 품질 평가

생성된 이미지들의 특징:

### 5.1 이미지 해상도
- **크기**: 1152 × 1024 pixels (config에서 설정된 downsampled 해상도)
- **포맷**: PNG (무손실)
- **파일 크기**: 평균 75-85KB (압축 후)

### 5.2 시각적 품질
생성된 샘플 이미지 분석 (수동 확인 필요):
- 3D Gaussian Splatting 렌더링 결과
- 학습된 모델 체크포인트 기반: `output/markerless_mouse_nerf/checkpoint.pt`
- 학습 메트릭: IoU ~0.87, SSIM ~0.97 (50 epochs)

---

## 6. 향후 작업 계획

### 6.1 즉시 실행 가능한 작업

#### A. GPU 메모리 최적화 후 재실행
```bash
# 1. 백그라운드 프로세스 종료
pkill -f train_script.py
pkill -f evaluate_model.py

# 2. GPU 메모리 확인
nvidia-smi

# 3. 시각화 재실행
source /home/joon/miniconda3/etc/profile.d/conda.sh
conda activate splatter
bash run_all_visualization.sh
```

#### B. FFmpeg 설치 및 비디오 생성
```bash
# x264 코덱 설치
conda install -c conda-forge x264 ffmpeg

# 비디오 생성 (이미지 파일들이 준비된 경우)
# Temporal sequence video
ffmpeg -framerate 30 -i output/markerless_mouse_nerf/renders/temporal/frame%04d.png \
       -c:v libx264 -pix_fmt yuv420p -crf 18 \
       output/markerless_mouse_nerf/renders/temporal/temporal_sequence.mp4

# 360-degree rotation video
ffmpeg -framerate 30 -i output/markerless_mouse_nerf/renders/rotation360/rot%03d.png \
       -c:v libx264 -pix_fmt yuv420p -crf 18 \
       output/markerless_mouse_nerf/renders/rotation360/rotation360.mp4
```

### 6.2 추가 개선사항

#### 렌더링 최적화
1. **메모리 효율 개선**
   - `render_image.py`에 `torch.cuda.empty_cache()` 추가
   - 각 렌더링 후 명시적으로 GPU 메모리 해제

2. **배치 처리**
   - 순차 실행 대신 메모리 여유에 따라 동적 배치 크기 조정

3. **체크포인트 재사용**
   - 모델 로딩을 한 번만 수행하고 여러 프레임 렌더링

#### 시각화 확장
1. **비교 시각화**
   - Ground truth vs. 렌더링 결과 side-by-side 비교
   - 오차 히트맵 생성

2. **인터랙티브 뷰어**
   - WebGL 기반 3D 포인트 클라우드 뷰어
   - Three.js 또는 Plotly 활용

3. **메트릭 오버레이**
   - 각 프레임에 SSIM, PSNR 값 표시
   - 카메라 파라미터 정보 오버레이

---

## 7. 결론

### 7.1 성과
- ✅ 4가지 시각화 방식 완전 구현
- ✅ torch_scatter 호환성 문제 해결
- ✅ 통합 실행 스크립트 작성
- ✅ 부분적 시각화 결과 생성 (24개 이미지)

### 7.2 제약사항
- ⚠️ GPU 메모리 부족으로 전체 데이터셋의 일부만 생성
- ⚠️ FFmpeg 인코더 부재로 비디오 미생성
- ⚠️ 포인트 클라우드 export 미완료

### 7.3 권장사항
1. **GPU 리소스 확보**: 백그라운드 프로세스 정리 후 재실행
2. **FFmpeg 설치**: 비디오 생성을 위한 환경 구성
3. **점진적 렌더링**: 작은 배치로 나누어 순차 실행

---

## 8. 참고 자료

### 생성된 파일
- `generate_multiview.py` - 멀티뷰 렌더링 스크립트
- `generate_temporal_video.py` - 시간순서 시퀀스 생성
- `generate_360_rotation.py` - 360도 회전 뷰 생성
- `export_point_cloud.py` - 3D 포인트 클라우드 export
- `run_all_visualization.sh` - 통합 실행 스크립트
- `monitor_visualization.sh` - 진행상황 모니터링 스크립트

### 로그 파일
- `output/markerless_mouse_nerf/logs/vis_full_pipeline.log` - 전체 실행 로그
- `output/markerless_mouse_nerf/logs/vis_multiview.log` - 멀티뷰 로그
- `output/markerless_mouse_nerf/logs/visualization.log` - 이전 실행 로그

### 모델 체크포인트
- `output/markerless_mouse_nerf/checkpoint.pt` - 학습된 모델 (50 epochs)
- 학습 메트릭: IoU 0.8712, SSIM 0.9744

---

**보고서 작성 완료**
다음 작업: GPU 메모리 확보 후 전체 시각화 재실행 권장
