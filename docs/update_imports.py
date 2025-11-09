#!/usr/bin/env python3
"""
Pose Splatter Import 경로 자동 업데이트 스크립트

리팩토링 후 모든 Python 파일의 import 문을 새 구조에 맞게 업데이트합니다.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Import 경로 매핑 (old_pattern → new_replacement)
IMPORT_MAPPINGS = [
    # src/ 모듈 경로 변경
    (r'from src\.model import', 'from src.core.model import'),
    (r'from src\.data import', 'from src.core.data import'),
    (r'from src\.unet_3d import', 'from src.core.unet_3d import'),
    (r'import src\.model', 'import src.core.model'),
    (r'import src\.data', 'import src.core.data'),
    (r'import src\.unet_3d', 'import src.core.unet_3d'),

    # Preprocessing
    (r'from src\.shape_carving import', 'from src.preprocessing.shape_carving import'),
    (r'from src\.shape_carver import', 'from src.preprocessing.shape_carving import'),
    (r'import src\.shape_carving', 'import src.preprocessing.shape_carving'),
    (r'import src\.shape_carver', 'import src.preprocessing.shape_carving'),

    # Utils
    (r'from src\.config_utils import', 'from src.utils.config_utils import'),
    (r'from src\.tracking import', 'from src.utils.tracking import'),
    (r'from src\.plots import', 'from src.utils.plots import'),
    (r'from src\.utils import', 'from src.utils.general import'),
    (r'import src\.config_utils', 'import src.utils.config_utils'),
    (r'import src\.tracking', 'import src.utils.tracking'),
    (r'import src\.plots', 'import src.utils.plots'),
    (r'import src\.utils', 'import src.utils.general'),

    # 루트 레벨 모듈 (스크립트 간 import가 있다면)
    (r'import estimate_up_direction', 'import scripts.pipeline.step1_estimate_up'),
    (r'import calculate_center_rotation', 'import scripts.pipeline.step2_center_rotation'),
    (r'import train_script', 'import scripts.training.train'),
    (r'import evaluate_model', 'import scripts.training.evaluate'),
]

def find_python_files(root_dir: Path) -> List[Path]:
    """모든 Python 파일 찾기 (output, data, __pycache__ 제외)"""
    python_files = []
    exclude_dirs = {'output', 'data', '__pycache__', '.git', 'venv', 'env'}

    for path in root_dir.rglob('*.py'):
        # 제외 디렉토리 체크
        if any(exclude_dir in path.parts for exclude_dir in exclude_dirs):
            continue
        python_files.append(path)

    return python_files

def update_imports_in_file(file_path: Path, mappings: List[Tuple[str, str]]) -> int:
    """파일의 import 문을 업데이트하고 변경 횟수 반환"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_count = 0

        # 각 매핑 패턴 적용
        for old_pattern, new_replacement in mappings:
            new_content, count = re.subn(old_pattern, new_replacement, content)
            if count > 0:
                print(f"  {file_path.relative_to(Path.cwd())}: {old_pattern} → {new_replacement} ({count}회)")
                changes_count += count
                content = new_content

        # 변경사항이 있으면 파일 업데이트
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        return changes_count

    except Exception as e:
        print(f"⚠️  오류 발생 ({file_path}): {e}")
        return 0

def main():
    print("=" * 60)
    print("Pose Splatter Import 경로 자동 업데이트")
    print("=" * 60)
    print()

    root_dir = Path.cwd()

    # Python 파일 찾기
    print("📂 Python 파일 검색 중...")
    python_files = find_python_files(root_dir)
    print(f"✓ {len(python_files)}개 파일 발견")
    print()

    # Import 업데이트
    print("🔄 Import 경로 업데이트 중...")
    print()

    total_changes = 0
    files_changed = 0

    for file_path in python_files:
        changes = update_imports_in_file(file_path, IMPORT_MAPPINGS)
        if changes > 0:
            total_changes += changes
            files_changed += 1

    print()
    print("=" * 60)
    print("✅ 업데이트 완료")
    print("=" * 60)
    print(f"수정된 파일: {files_changed}개")
    print(f"총 변경 횟수: {total_changes}회")
    print()

    if files_changed > 0:
        print("다음 단계:")
        print("1. 변경사항 검토: git diff")
        print("2. 테스트 실행: 각 스크립트 실행 확인")
        print("3. Git commit: git add . && git commit -m \"Update import paths after refactoring\"")
    else:
        print("변경사항 없음 (이미 업데이트되었거나 import가 없음)")

if __name__ == '__main__':
    main()
