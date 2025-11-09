# Pose Splatter 리팩토링 계획서

**작성일**: 2025-11-09
**우선순위**: 높음
**실행 시점**: 현재 훈련 완료 후

---

## 📋 목적

현재 프로젝트 구조를 체계화하여:
1. 코드 가독성 및 유지보수성 향상
2. 모듈 간 의존성 명확화
3. 새로운 기능 추가 용이성 확보
4. 재사용 가능한 컴포넌트 분리

---

## 📊 현재 프로젝트 구조 분석

### 현재 상태 (Before)

```
pose-splatter/
├── README.md                           # 메인 문서
├── LICENSE.md                          # 라이선스
├── requirements.txt                    # 패키지 의존성
├── environment.yml                     # Conda 환경
│
├── configs/                            # 설정 파일 (14개)
│   ├── markerless_mouse_nerf.json
│   ├── markerless_mouse_nerf_high_res.json
│   ├── markerless_mouse_nerf_fast.json
│   └── ...
│
├── src/                                # 핵심 라이브러리 (9개 모듈)
│   ├── config_utils.py                # 설정 유틸리티
│   ├── data.py                        # 데이터 로더
│   ├── model.py                       # 3D Gaussian 모델
│   ├── shape_carver.py                # Shape carving
│   ├── shape_carving.py               # Shape carving (중복?)
│   ├── tracking.py                    # 트래킹 유틸
│   ├── unet_3d.py                     # 3D U-Net
│   ├── utils.py                       # 일반 유틸리티
│   └── plots.py                       # 플롯 유틸리티
│
├── docs/                               # 문서 (이동 완료)
│   ├── reports/
│   │   ├── 251109_experiment_baseline.md
│   │   ├── 251109_execution_summary.md
│   │   ├── ANALYSIS_GUIDE.md
│   │   └── TOOLS_SUMMARY.md
│   ├── ANALYSIS_REPORT.md
│   ├── EXECUTION_SUMMARY.md
│   ├── SETUP_GUIDE.md
│   ├── QUICKSTART.md
│   └── README_ENHANCED.md
│
├── assets/                             # 이미지 리소스
│   └── teaser.png
│
└── [ROOT 17개 Python 스크립트]         # **정리 필요**
    ├── 1. Pipeline Scripts (6개)
    │   ├── estimate_up_direction.py
    │   ├── auto_estimate_up.py
    │   ├── calculate_center_rotation.py
    │   ├── calculate_crop_indices.py
    │   ├── write_images.py
    │   └── copy_to_zarr.py
    │
    ├── 2. Training & Inference (3개)
    │   ├── train_script.py
    │   ├── evaluate_model.py
    │   └── render_image.py
    │
    ├── 3. Feature Extraction (2개)
    │   ├── calculate_visual_features.py
    │   └── calculate_visual_embedding.py
    │
    ├── 4. Analysis & Visualization (4개)  # **새로 추가됨**
    │   ├── analyze_results.py
    │   ├── visualize_training.py
    │   ├── visualize_renders.py
    │   └── compare_configs.py
    │
    └── 5. Utilities (2개)
        ├── convert_camera_params.py
        └── plot_voxels.py
```

### 문제점

1. **루트 디렉토리 혼잡**: 17개의 Python 스크립트가 루트에 평면적으로 위치
2. **모듈 분류 불명확**: 용도별 그룹핑 없음
3. **중복 가능성**: `shape_carver.py`와 `shape_carving.py` 중복 확인 필요
4. **스크립트 vs 라이브러리 구분 부재**: 실행 스크립트와 재사용 모듈 혼재

---

## 🎯 제안하는 새로운 구조 (After)

```
pose-splatter/
├── README.md
├── LICENSE.md
├── requirements.txt
├── environment.yml
│
├── configs/                            # 설정 파일 (유지)
│
├── src/                                # 핵심 라이브러리 (리팩토링)
│   ├── __init__.py
│   ├── core/                          # 핵심 모델 및 데이터
│   │   ├── __init__.py
│   │   ├── model.py                   # 3D Gaussian Splat 모델
│   │   ├── data.py                    # 데이터 로더
│   │   └── unet_3d.py                 # 3D U-Net 모델
│   │
│   ├── preprocessing/                  # 전처리 모듈
│   │   ├── __init__.py
│   │   ├── shape_carving.py           # Shape carving (통합)
│   │   ├── camera_utils.py            # 카메라 변환 유틸
│   │   └── volume_processing.py       # 볼륨 처리
│   │
│   ├── training/                       # 훈련 관련
│   │   ├── __init__.py
│   │   ├── trainer.py                 # 훈련 로직
│   │   └── losses.py                  # 손실 함수
│   │
│   ├── evaluation/                     # 평가 관련
│   │   ├── __init__.py
│   │   ├── metrics.py                 # 메트릭 계산
│   │   └── renderer.py                # 렌더링 유틸
│   │
│   ├── analysis/                       # 분석 및 시각화
│   │   ├── __init__.py
│   │   ├── result_analyzer.py         # 결과 분석
│   │   ├── training_visualizer.py     # 훈련 시각화
│   │   └── render_visualizer.py       # 렌더링 시각화
│   │
│   └── utils/                          # 일반 유틸리티
│       ├── __init__.py
│       ├── config_utils.py            # 설정 로드
│       ├── tracking.py                # 트래킹
│       └── plots.py                   # 플롯 헬퍼
│
├── scripts/                            # 실행 스크립트 (새로 생성)
│   ├── pipeline/                      # 파이프라인 단계별 스크립트
│   │   ├── step1_estimate_up.py
│   │   ├── step2_center_rotation.py
│   │   ├── step3_crop_indices.py
│   │   ├── step4_write_images.py
│   │   └── step5_copy_to_zarr.py
│   │
│   ├── training/                      # 훈련 및 평가 스크립트
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── render.py
│   │
│   ├── analysis/                      # 분석 스크립트
│   │   ├── analyze_results.py
│   │   ├── visualize_training.py
│   │   ├── visualize_renders.py
│   │   └── compare_configs.py
│   │
│   ├── features/                      # 특징 추출 스크립트
│   │   ├── calculate_visual_features.py
│   │   └── calculate_visual_embedding.py
│   │
│   └── utils/                         # 유틸리티 스크립트
│       ├── convert_camera_params.py
│       └── plot_voxels.py
│
├── tools/                              # 자동화 도구 (shell scripts)
│   ├── run_full_pipeline.sh
│   ├── run_pipeline_auto.sh
│   └── monitor_pipeline.sh
│
├── docs/                               # 문서
│   ├── reports/
│   ├── REFACTORING_PLAN.md           # 이 문서
│   └── ...
│
└── assets/                             # 리소스
    └── teaser.png
```

---

## 🔄 마이그레이션 맵 (파일별 이동 계획)

### Phase 1: 스크립트 재배치

| 현재 위치 (ROOT) | 새 위치 | 카테고리 |
|------------------|---------|----------|
| `estimate_up_direction.py` | `scripts/pipeline/step1_estimate_up.py` | Pipeline |
| `auto_estimate_up.py` | `scripts/pipeline/step1_auto_estimate_up.py` | Pipeline |
| `calculate_center_rotation.py` | `scripts/pipeline/step2_center_rotation.py` | Pipeline |
| `calculate_crop_indices.py` | `scripts/pipeline/step3_crop_indices.py` | Pipeline |
| `write_images.py` | `scripts/pipeline/step4_write_images.py` | Pipeline |
| `copy_to_zarr.py` | `scripts/pipeline/step5_copy_to_zarr.py` | Pipeline |
| `train_script.py` | `scripts/training/train.py` | Training |
| `evaluate_model.py` | `scripts/training/evaluate.py` | Training |
| `render_image.py` | `scripts/training/render.py` | Training |
| `analyze_results.py` | `scripts/analysis/analyze_results.py` | Analysis |
| `visualize_training.py` | `scripts/analysis/visualize_training.py` | Analysis |
| `visualize_renders.py` | `scripts/analysis/visualize_renders.py` | Analysis |
| `compare_configs.py` | `scripts/analysis/compare_configs.py` | Analysis |
| `calculate_visual_features.py` | `scripts/features/calculate_visual_features.py` | Features |
| `calculate_visual_embedding.py` | `scripts/features/calculate_visual_embedding.py` | Features |
| `convert_camera_params.py` | `scripts/utils/convert_camera_params.py` | Utils |
| `plot_voxels.py` | `scripts/utils/plot_voxels.py` | Utils |

### Phase 2: src/ 모듈 리팩토링

| 현재 위치 (src/) | 새 위치 | 변경 사항 |
|-----------------|---------|----------|
| `model.py` | `src/core/model.py` | 이동 |
| `data.py` | `src/core/data.py` | 이동 |
| `unet_3d.py` | `src/core/unet_3d.py` | 이동 |
| `shape_carver.py` + `shape_carving.py` | `src/preprocessing/shape_carving.py` | **통합 필요** |
| - | `src/preprocessing/camera_utils.py` | **신규 생성** (카메라 관련 추출) |
| - | `src/preprocessing/volume_processing.py` | **신규 생성** (볼륨 관련 추출) |
| - | `src/training/trainer.py` | **신규 생성** (train_script.py에서 추출) |
| - | `src/training/losses.py` | **신규 생성** (loss 함수 분리) |
| - | `src/evaluation/metrics.py` | **신규 생성** (evaluate_model.py에서 추출) |
| - | `src/evaluation/renderer.py` | **신규 생성** (render_image.py에서 추출) |
| - | `src/analysis/result_analyzer.py` | **신규 생성** (analyze_results.py 로직 추출) |
| - | `src/analysis/training_visualizer.py` | **신규 생성** (visualize_training.py 로직 추출) |
| - | `src/analysis/render_visualizer.py` | **신규 생성** (visualize_renders.py 로직 추출) |
| `config_utils.py` | `src/utils/config_utils.py` | 이동 |
| `tracking.py` | `src/utils/tracking.py` | 이동 |
| `plots.py` | `src/utils/plots.py` | 이동 |
| `utils.py` | `src/utils/general.py` | 이름 변경 (명확성) |

### Phase 3: 자동화 스크립트 이동

| 현재 위치 (ROOT) | 새 위치 |
|-----------------|---------|
| `run_full_pipeline.sh` | `tools/run_full_pipeline.sh` |
| `run_pipeline_auto.sh` | `tools/run_pipeline_auto.sh` |
| `monitor_pipeline.sh` | `tools/monitor_pipeline.sh` |

---

## 📝 리팩토링 단계별 실행 계획

### Phase 1: 준비 단계 (훈련 완료 전)

**이미 완료**:
- [x] 문서 파일 docs로 이동
- [x] 리팩토링 계획서 작성

**추가 준비**:
- [ ] src/ 모듈 중복 확인 (`shape_carver.py` vs `shape_carving.py`)
- [ ] 전체 import 의존성 분석
- [ ] 테스트 케이스 확인 (있다면)

---

### Phase 2: 구조 변경 (훈련 완료 후 즉시 실행)

#### Step 1: 디렉토리 구조 생성
```bash
mkdir -p scripts/{pipeline,training,analysis,features,utils}
mkdir -p src/{core,preprocessing,training,evaluation,analysis,utils}
mkdir -p tools
```

#### Step 2: 스크립트 파일 이동 및 이름 변경
```bash
# Pipeline scripts
mv estimate_up_direction.py scripts/pipeline/step1_estimate_up.py
mv auto_estimate_up.py scripts/pipeline/step1_auto_estimate_up.py
mv calculate_center_rotation.py scripts/pipeline/step2_center_rotation.py
mv calculate_crop_indices.py scripts/pipeline/step3_crop_indices.py
mv write_images.py scripts/pipeline/step4_write_images.py
mv copy_to_zarr.py scripts/pipeline/step5_copy_to_zarr.py

# Training scripts
mv train_script.py scripts/training/train.py
mv evaluate_model.py scripts/training/evaluate.py
mv render_image.py scripts/training/render.py

# Analysis scripts
mv analyze_results.py scripts/analysis/analyze_results.py
mv visualize_training.py scripts/analysis/visualize_training.py
mv visualize_renders.py scripts/analysis/visualize_renders.py
mv compare_configs.py scripts/analysis/compare_configs.py

# Feature scripts
mv calculate_visual_features.py scripts/features/calculate_visual_features.py
mv calculate_visual_embedding.py scripts/features/calculate_visual_embedding.py

# Utility scripts
mv convert_camera_params.py scripts/utils/convert_camera_params.py
mv plot_voxels.py scripts/utils/plot_voxels.py

# Shell scripts
mv run_full_pipeline.sh tools/run_full_pipeline.sh
mv run_pipeline_auto.sh tools/run_pipeline_auto.sh
mv monitor_pipeline.sh tools/monitor_pipeline.sh
chmod +x tools/*.sh
```

#### Step 3: src/ 모듈 재구성
```bash
# Core modules
mv src/model.py src/core/model.py
mv src/data.py src/core/data.py
mv src/unet_3d.py src/core/unet_3d.py

# Preprocessing (통합 필요 - 수동 작업)
# shape_carver.py와 shape_carving.py 비교 후 통합

# Utils
mv src/config_utils.py src/utils/config_utils.py
mv src/tracking.py src/utils/tracking.py
mv src/plots.py src/utils/plots.py
mv src/utils.py src/utils/general.py
```

#### Step 4: __init__.py 파일 생성
각 새 디렉토리에 `__init__.py` 생성하여 패키지화

---

### Phase 3: Import 경로 업데이트

모든 스크립트의 import 문을 새 구조에 맞게 수정:

**Before**:
```python
from src.model import GaussianSplattingModel
from src.data import PoseSplatterDataset
import config_utils
```

**After**:
```python
from src.core.model import GaussianSplattingModel
from src.core.data import PoseSplatterDataset
from src.utils.config_utils import load_config
```

**자동화 스크립트 예시**:
```bash
# 일괄 import 경로 업데이트 (신중하게 실행)
find scripts -name "*.py" -exec sed -i 's/from src.model import/from src.core.model import/g' {} +
find scripts -name "*.py" -exec sed -i 's/from src.data import/from src.core.data import/g' {} +
```

---

### Phase 4: 문서 업데이트

- [ ] README.md 업데이트 (새 디렉토리 구조 반영)
- [ ] 모든 문서의 스크립트 경로 수정
- [ ] ANALYSIS_GUIDE.md 업데이트
- [ ] TOOLS_SUMMARY.md 업데이트

**Before**:
```bash
python3 train_script.py configs/markerless_mouse_nerf.json --epochs 50
```

**After**:
```bash
python3 scripts/training/train.py configs/markerless_mouse_nerf.json --epochs 50
```

---

### Phase 5: 테스트 및 검증

- [ ] 각 스크립트 개별 실행 테스트
- [ ] 전체 파이프라인 통합 테스트
- [ ] Import 오류 확인 및 수정
- [ ] 문서화된 모든 예제 명령어 실행 확인

---

## 🔍 주요 리팩토링 포인트

### 1. shape_carver.py vs shape_carving.py 통합

**조사 필요**:
- 두 파일의 기능 비교
- 중복 코드 확인
- 하나로 통합 가능 여부

**통합 후 위치**: `src/preprocessing/shape_carving.py`

### 2. 스크립트에서 라이브러리 로직 분리

**현재 문제**: `train_script.py`, `evaluate_model.py` 등이 실행 로직과 핵심 로직 혼재

**리팩토링 방향**:
- 핵심 로직 → `src/training/trainer.py`, `src/evaluation/metrics.py`로 이동
- 스크립트는 CLI 인터페이스와 설정만 담당

**예시**:

**Before** (`train_script.py`):
```python
# 200줄의 훈련 로직 + argparse + main
def main():
    # 모든 훈련 로직이 여기에
    ...
```

**After**:

`src/training/trainer.py`:
```python
class PoseSplatterTrainer:
    def __init__(self, config):
        ...

    def train(self):
        # 핵심 훈련 로직
        ...
```

`scripts/training/train.py`:
```python
from src.training.trainer import PoseSplatterTrainer
from src.utils.config_utils import load_config

def main():
    config = load_config(args.config)
    trainer = PoseSplatterTrainer(config)
    trainer.train()
```

### 3. 분석 도구 모듈화

**현재**: `analyze_results.py`, `visualize_*.py` 등이 독립 스크립트

**리팩토링 후**:
- 재사용 가능한 분석 클래스 → `src/analysis/`
- CLI 인터페이스 → `scripts/analysis/`

---

## 🚨 주의사항 및 리스크

### 1. Import 경로 변경
- **리스크**: 모든 파일의 import 문을 정확히 수정하지 않으면 런타임 에러
- **완화**: 자동화 스크립트 + 수동 검증

### 2. 실행 경로 변경
- **리스크**: 쉘 스크립트, 문서의 모든 경로 업데이트 필요
- **완화**: grep으로 모든 참조 검색 후 일괄 수정

### 3. 기존 실험 재현성
- **리스크**: 구조 변경으로 기존 체크포인트나 로그 접근 불가
- **완화**: output/ 디렉토리는 건드리지 않음, 상대 경로 유지

### 4. 협업 충돌
- **리스크**: 다른 개발자가 작업 중이라면 큰 충돌 발생
- **완화**: 훈련 완료 후 한 번에 실행, Git branch 사용

---

## ✅ 체크리스트

### 리팩토링 실행 전
- [ ] 현재 훈련 완료 대기
- [ ] Git에 현재 상태 커밋 (백업)
- [ ] 새 브랜치 생성 (`git checkout -b refactor-structure`)
- [ ] shape_carver vs shape_carving 중복 확인
- [ ] 전체 import 의존성 맵 생성

### 리팩토링 실행 중
- [ ] Phase 2 Step 1-4 순차 실행
- [ ] 각 단계마다 Git commit
- [ ] Import 경로 일괄 업데이트
- [ ] __init__.py 파일 생성

### 리팩토링 완료 후
- [ ] 각 스크립트 개별 실행 테스트
- [ ] 전체 파이프라인 dry-run
- [ ] 문서 모두 업데이트 (README, guides)
- [ ] Git commit 및 PR 생성
- [ ] 기존 main 브랜치 백업 태그 생성

---

## 📦 예상 결과

### 리팩토링 후 장점

1. **명확한 구조**:
   - 실행 스크립트 (`scripts/`)와 라이브러리 코드 (`src/`) 명확히 분리
   - 기능별 디렉토리로 빠른 탐색 가능

2. **재사용성 향상**:
   - `src/` 모듈을 다른 프로젝트에서 import 가능
   - 분석 도구를 라이브러리로 사용 가능

3. **유지보수 용이**:
   - 특정 기능 수정 시 해당 모듈만 접근
   - 새 파이프라인 단계 추가 간편

4. **확장성**:
   - 새로운 분석 도구 추가 → `scripts/analysis/`
   - 새로운 모델 추가 → `src/core/`

### 성능 영향
- **없음**: 구조 변경만이므로 실행 속도 동일
- Import 경로만 변경, 코드 로직은 유지

---

## 📅 타임라인

| 단계 | 예상 소요 시간 | 실행 시점 |
|------|---------------|----------|
| Phase 1 (준비) | 1-2시간 | 훈련 완료 전 (대기 중) |
| Phase 2 (구조 변경) | 2-3시간 | 훈련 완료 직후 |
| Phase 3 (Import 업데이트) | 3-4시간 | Phase 2 직후 |
| Phase 4 (문서 업데이트) | 2-3시간 | Phase 3 직후 |
| Phase 5 (테스트) | 2-3시간 | 최종 단계 |
| **총 예상 시간** | **10-15시간** | **1-2일** |

---

## 🔗 참고 자료

- Python 프로젝트 구조 Best Practices: [Real Python](https://realpython.com/python-application-layouts/)
- Import 시스템 이해: [Python Docs](https://docs.python.org/3/reference/import.html)
- 기존 프로젝트 문서: `docs/ANALYSIS_GUIDE.md`, `docs/TOOLS_SUMMARY.md`

---

**작성자**: Claude Code
**검토 필요**: 훈련 완료 후 사용자 승인
**업데이트 이력**:
- 2025-11-09: 초안 작성
