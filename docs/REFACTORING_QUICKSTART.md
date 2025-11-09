# 리팩토링 빠른 시작 가이드

**목적**: 훈련 완료 후 프로젝트 구조를 체계적으로 정리
**소요 시간**: 약 30분 (자동화 스크립트 사용)
**중요도**: ⭐⭐⭐ (높음)

---

## 🚀 빠른 실행 (3단계)

### 1단계: 백업 생성 (필수!)

```bash
cd /home/joon/dev/pose-splatter

# 현재 상태 커밋
git add .
git commit -m "Before refactoring: Save current state"

# 백업 브랜치 생성 (선택사항이지만 권장)
git checkout -b refactor-structure
```

### 2단계: 리팩토링 실행

```bash
# 자동화 스크립트 실행
bash docs/refactor_execute.sh
```

**이 스크립트는**:
- 새 디렉토리 구조 생성 (`scripts/`, `tools/`, `src/` 하위)
- 모든 Python 파일을 적절한 위치로 이동
- `__init__.py` 파일 자동 생성
- Git으로 파일 이동 추적

### 3단계: Import 경로 업데이트

```bash
# Import 문 자동 수정
python3 docs/update_imports.py
```

**이 스크립트는**:
- 모든 Python 파일의 import 문 스캔
- 새 경로로 자동 업데이트 (예: `from src.model` → `from src.core.model`)
- 변경사항 요약 출력

---

## ✅ 실행 후 체크리스트

### 즉시 확인 (필수)

```bash
# 1. 변경사항 검토
git status
git diff

# 2. 간단한 import 테스트
python3 -c "from src.core.model import GaussianSplattingModel; print('✓ Import OK')"
python3 -c "from src.core.data import PoseSplatterDataset; print('✓ Import OK')"

# 3. 파이프라인 스크립트 경로 확인
ls scripts/pipeline/
ls scripts/training/
ls scripts/analysis/
```

### 기능 테스트 (중요)

```bash
# 훈련 스크립트 dry-run (실제 실행 안함, import만 확인)
python3 -c "import sys; sys.path.insert(0, '.'); import scripts.training.train"

# 분석 스크립트 확인
python3 scripts/analysis/analyze_results.py --help

# 전체 파이프라인 명령어 확인
cat tools/run_pipeline_auto.sh
```

### 문서 업데이트 필요

- [ ] `README.md` - 스크립트 경로 업데이트
- [ ] `docs/ANALYSIS_GUIDE.md` - 경로 업데이트
- [ ] `docs/TOOLS_SUMMARY.md` - 경로 업데이트
- [ ] `tools/run_pipeline_auto.sh` - 스크립트 경로 수정
- [ ] `tools/run_full_pipeline.sh` - 스크립트 경로 수정

---

## 📋 Before & After 비교

### 명령어 변경사항

| Before (기존) | After (리팩토링 후) |
|--------------|-------------------|
| `python3 train_script.py configs/...` | `python3 scripts/training/train.py configs/...` |
| `python3 evaluate_model.py configs/...` | `python3 scripts/training/evaluate.py configs/...` |
| `python3 analyze_results.py configs/...` | `python3 scripts/analysis/analyze_results.py configs/...` |
| `python3 visualize_training.py --log_file ...` | `python3 scripts/analysis/visualize_training.py --log_file ...` |
| `python3 calculate_center_rotation.py configs/...` | `python3 scripts/pipeline/step2_center_rotation.py configs/...` |
| `bash run_pipeline_auto.sh` | `bash tools/run_pipeline_auto.sh` |
| `bash monitor_pipeline.sh` | `bash tools/monitor_pipeline.sh` |

### Import 변경사항

**Before**:
```python
from src.model import GaussianSplattingModel
from src.data import PoseSplatterDataset
from src.config_utils import load_config
import src.utils
```

**After**:
```python
from src.core.model import GaussianSplattingModel
from src.core.data import PoseSplatterDataset
from src.utils.config_utils import load_config
from src.utils import general
```

---

## 🔧 문제 해결

### Q1: "ModuleNotFoundError: No module named 'src.model'"

**원인**: Import 경로 업데이트가 완료되지 않음

**해결**:
```bash
# Import 업데이트 스크립트 다시 실행
python3 docs/update_imports.py

# 또는 수동으로 해당 파일 수정
# from src.model import ...
# ↓
# from src.core.model import ...
```

### Q2: "FileNotFoundError: train_script.py not found"

**원인**: 쉘 스크립트나 문서의 경로가 업데이트되지 않음

**해결**:
```bash
# tools/run_pipeline_auto.sh 수정
# python3 train_script.py → python3 scripts/training/train.py
sed -i 's|python3 train_script.py|python3 scripts/training/train.py|g' tools/run_pipeline_auto.sh
```

### Q3: Git 충돌 발생

**원인**: 리팩토링 중 다른 작업이 진행됨

**해결**:
```bash
# 리팩토링 취소
git reset --hard HEAD

# 백업 브랜치로 돌아가서 다시 시작
git checkout master
git branch -D refactor-structure
```

---

## 📦 리팩토링 완료 후 커밋

모든 테스트가 통과하면:

```bash
# 모든 변경사항 스테이징
git add .

# 리팩토링 커밋
git commit -m "Refactor: Reorganize project structure

- Move scripts to scripts/ directory (pipeline, training, analysis, features, utils)
- Move shell scripts to tools/ directory
- Reorganize src/ modules (core, preprocessing, training, evaluation, analysis, utils)
- Update all import paths
- Add __init__.py files for proper package structure

This refactoring improves code organization and maintainability."

# 메인 브랜치에 병합 (리뷰 후)
git checkout master
git merge refactor-structure
```

---

## 🎯 예상 결과

리팩토링 후 다음과 같은 이점을 얻습니다:

1. **명확한 구조**
   - 실행 스크립트 (`scripts/`) vs 라이브러리 코드 (`src/`) 분리
   - 기능별 디렉토리로 빠른 탐색

2. **향상된 유지보수성**
   - 새 분석 도구 추가 → `scripts/analysis/`
   - 새 파이프라인 단계 추가 → `scripts/pipeline/`
   - 코어 모델 수정 → `src/core/`

3. **재사용 가능**
   - `src/` 모듈을 다른 프로젝트에서 import 가능
   - 분석 도구를 라이브러리처럼 사용 가능

4. **확장성**
   - 새 실험 variant 추가 간편
   - 플러그인 형태로 기능 추가 가능

---

## 📚 관련 문서

- **상세 계획서**: `docs/REFACTORING_PLAN.md` - 전체 리팩토링 설계 및 이유
- **실행 스크립트**: `docs/refactor_execute.sh` - 자동화 스크립트 소스
- **Import 업데이트**: `docs/update_imports.py` - Import 경로 수정 스크립트

---

**작성일**: 2025-11-09
**실행 권장 시점**: 현재 훈련 완료 후 즉시
**예상 소요 시간**: 30분 (자동화) + 1-2시간 (테스트 및 문서 업데이트)
