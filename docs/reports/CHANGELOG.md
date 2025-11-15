# Pose Splatter 변경 이력

## [2025-11-10] - 시각화 파이프라인 구현

### 추가된 기능

#### 1. 통합 시각화 파이프라인
- **파일**: `run_all_visualization.sh`
- **기능**: 4가지 시각화 유형을 순차적으로 실행하는 통합 스크립트
- **사용법**: `bash run_all_visualization.sh`

#### 2. 멀티뷰 렌더링
- **파일**: `generate_multiview.py`
- **기능**: 동일 프레임을 6개 카메라 각도에서 렌더링
- **출력**: `output/markerless_mouse_nerf/renders/multiview/frame0000_view{0-5}.png`
- **파라미터**:
  - `config_path`: 설정 파일 경로
  - `frame_num`: 렌더링할 프레임 번호 (기본값: 0)
  - `num_cameras`: 카메라 개수 (기본값: 6)

#### 3. 시간순서 비디오 생성
- **파일**: `generate_temporal_video.py`
- **기능**: 연속된 프레임을 렌더링하고 비디오로 변환
- **출력**:
  - 이미지: `output/markerless_mouse_nerf/renders/temporal/frame{0000-0029}.png`
  - 비디오: `output/markerless_mouse_nerf/renders/temporal/temporal_sequence.mp4`
- **파라미터**:
  - `view_num`: 카메라 뷰 번호 (기본값: 0)
  - `start_frame`: 시작 프레임 (기본값: 0)
  - `num_frames`: 프레임 개수 (기본값: 60)
  - `fps`: 비디오 프레임레이트 (기본값: 30)

#### 4. 360도 회전 뷰
- **파일**: `generate_360_rotation.py`
- **기능**: 단일 프레임을 중심으로 360도 회전하며 렌더링
- **출력**:
  - 이미지: `output/markerless_mouse_nerf/renders/rotation360/rot{000-023}.png`
  - 비디오: `output/markerless_mouse_nerf/renders/rotation360/rotation360.mp4`
- **파라미터**:
  - `frame_num`: 프레임 번호 (기본값: 0)
  - `view_num`: 기준 카메라 뷰 (기본값: 0)
  - `num_angles`: 회전 각도 개수 (기본값: 36)

#### 5. 3D 포인트 클라우드 Export
- **파일**: `export_point_cloud.py`
- **기능**: 3D Gaussian Splatting 결과를 PLY 형식으로 내보내기
- **출력**: `output/markerless_mouse_nerf/pointclouds/frame{XXXX}.ply`
- **포함 데이터**:
  - 3D 위치 (x, y, z)
  - RGB 색상 (red, green, blue)
  - 투명도 (alpha)
- **사용법**:
  ```bash
  python3 export_point_cloud.py --config configs/markerless_mouse_nerf.json --frame 0
  ```

#### 6. 시각화 모니터링 스크립트
- **파일**: `monitor_visualization.sh`
- **기능**: 실시간 시각화 진행상황 모니터링
- **사용법**: `bash monitor_visualization.sh`
- **표시 정보**:
  - 멀티뷰 렌더링 진행도 (X/6)
  - 시간순서 프레임 진행도 (X/31)
  - 360도 회전 진행도 (X/25)
  - 포인트 클라우드 생성 여부 (X/1)
  - 최근 로그 항목

### 해결된 문제

#### torch_scatter 호환성 문제
- **증상**: `OSError: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev`
- **원인**: PyTorch 2.6.0 + CUDA 12.4와 torch_scatter 버전 불일치
- **해결**:
  ```bash
  pip uninstall torch_scatter -y
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
  ```
- **결과**: torch-scatter 2.1.2+pt26cu124 설치 완료

#### Conda 환경 활성화 문제
- **증상**: `ModuleNotFoundError: No module named 'torch'`
- **원인**: bash 스크립트에서 conda 환경 미활성화
- **해결**: `run_all_visualization.sh`에 conda 활성화 추가
  ```bash
  source /home/joon/miniconda3/etc/profile.d/conda.sh
  conda activate splatter
  ```

### 알려진 제약사항

#### 1. GPU 메모리 부족
- **영향**: 대규모 렌더링 작업 시 OOM 에러 발생
- **현상**:
  - 멀티뷰: 6개 중 일부만 성공
  - 시간순서: 30개 중 일부만 성공
  - 포인트 클라우드: 실패 가능
- **해결 방안**:
  - 백그라운드 프로세스 종료 후 재실행
  - GPU 메모리 정리: `torch.cuda.empty_cache()` 추가
  - 배치 크기 축소

#### 2. FFmpeg libx264 인코더 부재
- **영향**: 비디오 생성 불가
- **현상**: `Unknown encoder 'libx264'` 에러
- **해결 방안**:
  ```bash
  conda install -c conda-forge x264 ffmpeg
  ```

### 문서 업데이트

#### 새로 추가된 문서
- **VISUALIZATION_REPORT.md**: 시각화 구현 상세 보고서
  - 구현 내용 및 기술적 문제 해결
  - 실행 결과 및 품질 평가
  - 향후 작업 계획

#### 업데이트된 문서
- **README.md**:
  - "Comprehensive Visualization Pipeline" 섹션 추가
  - 시각화 스크립트 사용법 추가
  - Recent Updates 섹션 추가
  - Documentation 섹션에 VISUALIZATION_REPORT.md 링크 추가

### 파일 구조 변경

```
pose-splatter/
├── generate_multiview.py          (NEW)
├── generate_temporal_video.py     (NEW)
├── generate_360_rotation.py       (NEW)
├── export_point_cloud.py          (NEW)
├── run_all_visualization.sh       (NEW)
├── monitor_visualization.sh       (NEW)
├── VISUALIZATION_REPORT.md        (NEW)
├── CHANGELOG.md                   (NEW)
├── README.md                      (UPDATED)
└── output/markerless_mouse_nerf/
    ├── renders/                   (NEW)
    │   ├── multiview/
    │   ├── temporal/
    │   └── rotation360/
    ├── pointclouds/               (NEW)
    └── logs/
        ├── vis_full_pipeline.log  (NEW)
        └── vis_multiview.log      (NEW)
```

### 실행 환경

- **Python**: 3.10
- **PyTorch**: 2.6.0+cu124
- **CUDA**: 12.4
- **torch_scatter**: 2.1.2+pt26cu124
- **GPU**: NVIDIA RTX 3090 (11.75 GB)

### 다음 단계

#### 즉시 실행 가능
1. FFmpeg 설치 및 비디오 생성 완료
2. GPU 메모리 확보 후 전체 시각화 재실행
3. 포인트 클라우드 생성 검증

#### 향후 개선사항
1. **메모리 최적화**:
   - 렌더링 스크립트에 `torch.cuda.empty_cache()` 추가
   - 배치 처리 방식 개선

2. **시각화 확장**:
   - Ground truth vs. 렌더링 결과 비교 시각화
   - 오차 히트맵 생성
   - 인터랙티브 3D 뷰어 구현

3. **성능 개선**:
   - 멀티프로세싱을 활용한 병렬 렌더링
   - 체크포인트 재사용으로 로딩 시간 단축

### 참고사항

- 모든 시각화 스크립트는 학습된 모델 체크포인트 (`checkpoint.pt`) 필요
- 시각화 전 학습 완료 확인: IoU ~0.87, SSIM ~0.97 (50 epochs)
- 백그라운드 프로세스가 GPU 메모리를 점유 중이면 시각화 실패 가능
- 로그 파일은 `output/markerless_mouse_nerf/logs/` 디렉토리에 저장

---

## 기여자

- **구현**: Claude Code (Anthropic)
- **날짜**: 2025-11-10
- **버전**: v1.0.0
