# Pose Splatter - Quick Start Guide

## 전체 시퀀스 렌더링 및 시각화

### 1. 빠른 시각화 (30 프레임)

```bash
# 기본 시각화 파이프라인
bash scripts/visualization/run_all_visualization.sh
```

**생성 파일**:
- `output/markerless_mouse_nerf/renders/multiview/` - 6 카메라 뷰
- `output/markerless_mouse_nerf/renders/temporal/` - 30 프레임 + MP4
- `output/markerless_mouse_nerf/renders/rotation360/` - 360° 회전 + MP4

---

### 2. 전체 시퀀스 렌더링 (3600 프레임, ~2분)

#### 방법 A: Bash 스크립트 (이미지만)

```bash
# 전체 시퀀스 렌더링 (프레임 0-3600, 뷰 0)
bash scripts/visualization/render_full_sequence.sh \
  configs/baseline/markerless_mouse_nerf.json 0 3600 0

# 일부 프레임만 (프레임 0-1000)
bash scripts/visualization/render_full_sequence.sh \
  configs/baseline/markerless_mouse_nerf.json 0 1000 0
```

**출력**: `output/markerless_mouse_nerf/renders/full_sequence/`
- `frame00000.png` ~ `frame03599.png`
- `full_sequence.mp4` (자동 생성)

**예상 시간**:
- 3600 프레임: ~2-3시간 (GPU에 따라)
- 진행률: 5%마다 업데이트

#### 방법 B: Python 스크립트 (이미지 + Rerun)

```bash
# 전체 시퀀스 + Rerun 데이터 생성
python scripts/visualization/export_temporal_sequence_rerun.py \
  configs/baseline/markerless_mouse_nerf.json \
  --start 0 --end 3600 --view 0

# 일부만 (더 빠름)
python scripts/visualization/export_temporal_sequence_rerun.py \
  configs/baseline/markerless_mouse_nerf.json \
  --start 0 --end 500 --view 0
```

**출력**:
- PNG 이미지들
- `sequence.rrd` - Rerun 시각화 파일

---

### 3. Rerun 인터랙티브 시각화

#### 설치

```bash
pip install rerun-sdk
```

#### 사용법

```bash
# Rerun 뷰어 실행
rerun output/markerless_mouse_nerf/renders/full_sequence/sequence.rrd
```

**기능**:
- ⏯️ Timeline 재생/일시정지
- 🎬 프레임별 탐색
- 🎨 3D Gaussian 시각화
- 📷 멀티 뷰 (렌더링 + 3D 포인트)
- 🔍 확대/축소/회전

**단축키**:
- `Space`: 재생/일시정지
- `←/→`: 이전/다음 프레임
- `마우스 드래그`: 3D 뷰 회전
- `Scroll`: 확대/축소

---

### 4. 비디오 생성

#### 자동 (스크립트 실행 시)

렌더링 스크립트가 자동으로 MP4 생성 시도:
1. libx264 (최고 품질)
2. h264_nvenc (NVIDIA GPU 인코더)
3. GIF (fallback)

#### 수동 (렌더링 후)

```bash
# PNG 시퀀스에서 MP4 생성
bash scripts/visualization/create_videos.sh

# 또는 ffmpeg 직접 사용
ffmpeg -y -framerate 30 \
  -i output/markerless_mouse_nerf/renders/full_sequence/frame%05d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 \
  output/markerless_mouse_nerf/renders/full_sequence/full_sequence.mp4
```

**옵션**:
- `-framerate 30`: 30 FPS (원본 100 FPS → 3배 빠른 재생)
- `-framerate 15`: 15 FPS (6.7배 빠른 재생)
- `-crf 18`: 품질 (18=고품질, 23=보통, 28=낮음)

---

### 5. 사용 사례별 추천

#### Case 1: 빠른 확인 (5분)

```bash
# 30 프레임만
bash scripts/visualization/run_all_visualization.sh
```

#### Case 2: 전체 시퀀스 비디오 (2-3시간)

```bash
# 렌더링 + 비디오 자동 생성
bash scripts/visualization/render_full_sequence.sh \
  configs/baseline/markerless_mouse_nerf.json 0 3600 0
```

#### Case 3: 인터랙티브 3D 탐색 (2-3시간 + Rerun)

```bash
# 1. 렌더링 + Rerun 데이터 생성
python scripts/visualization/export_temporal_sequence_rerun.py \
  configs/baseline/markerless_mouse_nerf.json \
  --start 0 --end 3600 --view 0

# 2. Rerun 뷰어 실행
rerun output/markerless_mouse_nerf/renders/full_sequence/sequence.rrd
```

#### Case 4: 일부만 빠르게 (10-30분)

```bash
# 처음 500 프레임만
python scripts/visualization/export_temporal_sequence_rerun.py \
  configs/baseline/markerless_mouse_nerf.json \
  --start 0 --end 500 --view 0

# Rerun으로 확인
rerun output/markerless_mouse_nerf/renders/full_sequence/sequence.rrd
```

---

### 6. 멀티 뷰 렌더링

여러 카메라 뷰를 동시에 렌더링:

```bash
# 뷰 0, 2, 3 렌더링
for view in 0 2 3; do
    python scripts/visualization/export_temporal_sequence_rerun.py \
        configs/baseline/markerless_mouse_nerf.json \
        --start 0 --end 1000 --view $view \
        --output output/markerless_mouse_nerf/renders/view${view}
done
```

---

### 7. 백그라운드 실행

장시간 렌더링은 백그라운드로:

```bash
# nohup으로 백그라운드 실행
nohup bash scripts/visualization/render_full_sequence.sh \
  configs/baseline/markerless_mouse_nerf.json 0 3600 0 \
  > render.log 2>&1 &

# 진행 상황 모니터링
tail -f render.log

# 또는
watch -n 10 "ls output/markerless_mouse_nerf/renders/full_sequence/*.png | wc -l"
```

---

### 8. 저장 공간 관리

**전체 시퀀스 저장 공간**:
- PNG (3600 프레임): ~250-300 MB
- MP4 (libx264, CRF 18): ~20-30 MB
- Rerun .rrd: ~100-200 MB

**공간 절약**:
```bash
# PNG만 필요하면 Rerun 건너뛰기
python scripts/visualization/export_temporal_sequence_rerun.py \
  configs/baseline/markerless_mouse_nerf.json \
  --no-rerun

# 비디오만 필요하면 PNG 삭제
bash scripts/visualization/render_full_sequence.sh ...
rm output/markerless_mouse_nerf/renders/full_sequence/*.png
```

---

## 요약

| 목적 | 명령어 | 시간 | 출력 |
|------|--------|------|------|
| 빠른 확인 | `bash scripts/visualization/run_all_visualization.sh` | 5분 | 30 프레임 |
| 전체 비디오 | `bash scripts/visualization/render_full_sequence.sh ...` | 2-3시간 | 3600 PNG + MP4 |
| 3D 탐색 | `python scripts/visualization/export_temporal_sequence_rerun.py ...` | 2-3시간 | PNG + .rrd |
| Rerun 뷰어 | `rerun output/.../sequence.rrd` | 즉시 | 인터랙티브 |

---

## 문제 해결

**렌더링이 느림**:
- 프레임 범위 줄이기 (`--end 500`)
- 더 강력한 GPU 사용
- 백그라운드 실행

**비디오 생성 실패**:
```bash
# libx264 없으면 수동 설치
sudo apt-get install ffmpeg libx264-dev

# 또는 GIF로 대체
ffmpeg -framerate 15 -i frame%05d.png -vf "scale=576:-1" output.gif
```

**Rerun 설치 오류**:
```bash
pip install --upgrade rerun-sdk
```

**메모리 부족**:
- 프레임 범위 나누기 (0-1000, 1000-2000, ...)
- `--no-rerun` 옵션 사용
