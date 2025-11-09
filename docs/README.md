# Pose Splatter 문서 디렉토리

이 디렉토리는 Pose Splatter 프로젝트의 모든 문서를 포함합니다.

---

## 📚 문서 카테고리

### 🚀 시작 가이드

- **[QUICKSTART.md](QUICKSTART.md)** - 빠른 시작 가이드
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - 환경 설정 및 설치 가이드
- **[README_ENHANCED.md](README_ENHANCED.md)** - 확장된 사용 설명서

### 📊 실험 및 분석

- **[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** - 종합 분석 보고서
- **[EXECUTION_SUMMARY.md](EXECUTION_SUMMARY.md)** - 실행 요약
- **[reports/](reports/)** - 실험 보고서 폴더
  - `251109_experiment_baseline.md` - Baseline 실험 상세 보고서
  - `251109_execution_summary.md` - 실행 요약 (2025-11-09)
  - `ANALYSIS_GUIDE.md` - 분석 워크플로우 가이드
  - `TOOLS_SUMMARY.md` - 분석 도구 요약

### 🔧 리팩토링 (우선순위: 높음)

- **[REFACTORING_QUICKSTART.md](REFACTORING_QUICKSTART.md)** ⭐ **먼저 읽기**
  - 훈련 완료 후 즉시 실행할 리팩토링 빠른 가이드
  - 자동화 스크립트 사용법
  - 3단계로 완료 (백업 → 실행 → Import 업데이트)

- **[REFACTORING_PLAN.md](REFACTORING_PLAN.md)** - 상세 리팩토링 계획서
  - 현재 vs 제안하는 구조
  - 파일별 마이그레이션 맵
  - 단계별 실행 계획
  - 주의사항 및 리스크

- **[refactor_execute.sh](refactor_execute.sh)** - 리팩토링 자동화 스크립트
- **[update_imports.py](update_imports.py)** - Import 경로 자동 업데이트 스크립트

---

## 🎯 목적별 문서 찾기

### "처음 시작합니다"
→ [QUICKSTART.md](QUICKSTART.md) → [SETUP_GUIDE.md](SETUP_GUIDE.md)

### "실험 결과를 분석하고 싶습니다"
→ [reports/ANALYSIS_GUIDE.md](reports/ANALYSIS_GUIDE.md) → [reports/TOOLS_SUMMARY.md](reports/TOOLS_SUMMARY.md)

### "훈련이 완료되었습니다"
→ [REFACTORING_QUICKSTART.md](REFACTORING_QUICKSTART.md) ⭐ **우선 실행**

### "프로젝트 구조를 이해하고 싶습니다"
→ [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - "현재 프로젝트 구조 분석" 섹션

### "현재 실험 상태를 확인하고 싶습니다"
→ [reports/251109_execution_summary.md](reports/251109_execution_summary.md)

### "이전 실험 결과를 보고 싶습니다"
→ [reports/251109_experiment_baseline.md](reports/251109_experiment_baseline.md)

---

## 📋 문서 우선순위

### 🔴 높음 (훈련 완료 후 즉시)
1. [REFACTORING_QUICKSTART.md](REFACTORING_QUICKSTART.md) - 리팩토링 실행
2. [reports/ANALYSIS_GUIDE.md](reports/ANALYSIS_GUIDE.md) - 결과 분석

### 🟡 중간 (필요시)
3. [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - 구조 이해
4. [reports/TOOLS_SUMMARY.md](reports/TOOLS_SUMMARY.md) - 도구 활용
5. [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) - 종합 분석

### 🟢 낮음 (참고)
6. [README_ENHANCED.md](README_ENHANCED.md) - 확장 가이드
7. [EXECUTION_SUMMARY.md](EXECUTION_SUMMARY.md) - 실행 기록

---

## 🗂️ 디렉토리 구조

```
docs/
├── README.md                          # 이 파일
│
├── 시작 가이드/
│   ├── QUICKSTART.md                  # 빠른 시작
│   ├── SETUP_GUIDE.md                 # 환경 설정
│   └── README_ENHANCED.md             # 확장 가이드
│
├── 실험 및 분석/
│   ├── ANALYSIS_REPORT.md             # 종합 분석
│   ├── EXECUTION_SUMMARY.md           # 실행 요약
│   └── reports/                       # 실험 보고서
│       ├── 251109_experiment_baseline.md
│       ├── 251109_execution_summary.md
│       ├── ANALYSIS_GUIDE.md
│       └── TOOLS_SUMMARY.md
│
└── 리팩토링/ ⭐ 우선순위 높음
    ├── REFACTORING_QUICKSTART.md      # 빠른 가이드
    ├── REFACTORING_PLAN.md            # 상세 계획
    ├── refactor_execute.sh            # 실행 스크립트
    └── update_imports.py              # Import 업데이트
```

---

## 📝 문서 작성 규칙

### 파일명 컨벤션
- **UPPERCASE.md**: 주요 문서 (예: README.md, QUICKSTART.md)
- **YYMMDD_*.md**: 날짜별 보고서 (예: 251109_experiment_baseline.md)
- **snake_case.sh/py**: 실행 가능한 스크립트

### 문서 구조
- 명확한 제목과 섹션
- 실행 가능한 코드 블록
- Before/After 예시
- 체크리스트 포함

### 업데이트 규칙
- 각 문서 하단에 "작성일", "마지막 업데이트" 명시
- 주요 변경사항은 문서 상단에 "업데이트 이력" 추가

---

## 🔄 다음 업데이트 예정

리팩토링 완료 후:
- [ ] 모든 문서의 스크립트 경로 업데이트
- [ ] 새 디렉토리 구조 반영
- [ ] Before/After 예시 추가
- [ ] 스크린샷 추가 (선택)

---

**마지막 업데이트**: 2025-11-09
**관리자**: Claude Code
**문의**: 프로젝트 README.md 참조
