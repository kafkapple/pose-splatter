#!/bin/bash
# Pose Splatter 리팩토링 실행 스크립트
# 실행 전 반드시 백업 생성 권장: git commit -am "Before refactoring"

set -e  # 에러 발생 시 즉시 중단

echo "=========================================="
echo "Pose Splatter 구조 리팩토링 시작"
echo "=========================================="

# 현재 디렉토리 확인
if [ ! -f "README.md" ]; then
    echo "Error: 프로젝트 루트 디렉토리에서 실행해주세요"
    exit 1
fi

# 백업 확인
echo ""
echo "⚠️  경고: 이 스크립트는 프로젝트 구조를 크게 변경합니다."
echo "계속하기 전에 Git commit으로 백업했는지 확인하세요."
read -p "백업을 완료했습니까? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "리팩토링 취소됨"
    exit 0
fi

echo ""
echo "=========================================="
echo "Phase 1: 디렉토리 구조 생성"
echo "=========================================="

mkdir -p scripts/{pipeline,training,analysis,features,utils}
mkdir -p src/{core,preprocessing,training,evaluation,analysis,utils}
mkdir -p tools

echo "✓ 디렉토리 생성 완료"

echo ""
echo "=========================================="
echo "Phase 2: 파이프라인 스크립트 이동"
echo "=========================================="

# Pipeline scripts
if [ -f "estimate_up_direction.py" ]; then
    git mv estimate_up_direction.py scripts/pipeline/step1_estimate_up.py
    echo "✓ estimate_up_direction.py → scripts/pipeline/step1_estimate_up.py"
fi

if [ -f "auto_estimate_up.py" ]; then
    git mv auto_estimate_up.py scripts/pipeline/step1_auto_estimate_up.py
    echo "✓ auto_estimate_up.py → scripts/pipeline/step1_auto_estimate_up.py"
fi

if [ -f "calculate_center_rotation.py" ]; then
    git mv calculate_center_rotation.py scripts/pipeline/step2_center_rotation.py
    echo "✓ calculate_center_rotation.py → scripts/pipeline/step2_center_rotation.py"
fi

if [ -f "calculate_crop_indices.py" ]; then
    git mv calculate_crop_indices.py scripts/pipeline/step3_crop_indices.py
    echo "✓ calculate_crop_indices.py → scripts/pipeline/step3_crop_indices.py"
fi

if [ -f "write_images.py" ]; then
    git mv write_images.py scripts/pipeline/step4_write_images.py
    echo "✓ write_images.py → scripts/pipeline/step4_write_images.py"
fi

if [ -f "copy_to_zarr.py" ]; then
    git mv copy_to_zarr.py scripts/pipeline/step5_copy_to_zarr.py
    echo "✓ copy_to_zarr.py → scripts/pipeline/step5_copy_to_zarr.py"
fi

echo ""
echo "=========================================="
echo "Phase 3: 훈련/평가 스크립트 이동"
echo "=========================================="

if [ -f "train_script.py" ]; then
    git mv train_script.py scripts/training/train.py
    echo "✓ train_script.py → scripts/training/train.py"
fi

if [ -f "evaluate_model.py" ]; then
    git mv evaluate_model.py scripts/training/evaluate.py
    echo "✓ evaluate_model.py → scripts/training/evaluate.py"
fi

if [ -f "render_image.py" ]; then
    git mv render_image.py scripts/training/render.py
    echo "✓ render_image.py → scripts/training/render.py"
fi

echo ""
echo "=========================================="
echo "Phase 4: 분석 스크립트 이동"
echo "=========================================="

if [ -f "analyze_results.py" ]; then
    git mv analyze_results.py scripts/analysis/analyze_results.py
    echo "✓ analyze_results.py → scripts/analysis/analyze_results.py"
fi

if [ -f "visualize_training.py" ]; then
    git mv visualize_training.py scripts/analysis/visualize_training.py
    echo "✓ visualize_training.py → scripts/analysis/visualize_training.py"
fi

if [ -f "visualize_renders.py" ]; then
    git mv visualize_renders.py scripts/analysis/visualize_renders.py
    echo "✓ visualize_renders.py → scripts/analysis/visualize_renders.py"
fi

if [ -f "compare_configs.py" ]; then
    git mv compare_configs.py scripts/analysis/compare_configs.py
    echo "✓ compare_configs.py → scripts/analysis/compare_configs.py"
fi

echo ""
echo "=========================================="
echo "Phase 5: 특징 추출 스크립트 이동"
echo "=========================================="

if [ -f "calculate_visual_features.py" ]; then
    git mv calculate_visual_features.py scripts/features/calculate_visual_features.py
    echo "✓ calculate_visual_features.py → scripts/features/calculate_visual_features.py"
fi

if [ -f "calculate_visual_embedding.py" ]; then
    git mv calculate_visual_embedding.py scripts/features/calculate_visual_embedding.py
    echo "✓ calculate_visual_embedding.py → scripts/features/calculate_visual_embedding.py"
fi

echo ""
echo "=========================================="
echo "Phase 6: 유틸리티 스크립트 이동"
echo "=========================================="

if [ -f "convert_camera_params.py" ]; then
    git mv convert_camera_params.py scripts/utils/convert_camera_params.py
    echo "✓ convert_camera_params.py → scripts/utils/convert_camera_params.py"
fi

if [ -f "plot_voxels.py" ]; then
    git mv plot_voxels.py scripts/utils/plot_voxels.py
    echo "✓ plot_voxels.py → scripts/utils/plot_voxels.py"
fi

echo ""
echo "=========================================="
echo "Phase 7: 셸 스크립트 이동"
echo "=========================================="

if [ -f "run_full_pipeline.sh" ]; then
    git mv run_full_pipeline.sh tools/run_full_pipeline.sh
    chmod +x tools/run_full_pipeline.sh
    echo "✓ run_full_pipeline.sh → tools/run_full_pipeline.sh"
fi

if [ -f "run_pipeline_auto.sh" ]; then
    git mv run_pipeline_auto.sh tools/run_pipeline_auto.sh
    chmod +x tools/run_pipeline_auto.sh
    echo "✓ run_pipeline_auto.sh → tools/run_pipeline_auto.sh"
fi

if [ -f "monitor_pipeline.sh" ]; then
    git mv monitor_pipeline.sh tools/monitor_pipeline.sh
    chmod +x tools/monitor_pipeline.sh
    echo "✓ monitor_pipeline.sh → tools/monitor_pipeline.sh"
fi

echo ""
echo "=========================================="
echo "Phase 8: src/ 모듈 재구성"
echo "=========================================="

# Core modules
if [ -f "src/model.py" ]; then
    git mv src/model.py src/core/model.py
    echo "✓ src/model.py → src/core/model.py"
fi

if [ -f "src/data.py" ]; then
    git mv src/data.py src/core/data.py
    echo "✓ src/data.py → src/core/data.py"
fi

if [ -f "src/unet_3d.py" ]; then
    git mv src/unet_3d.py src/core/unet_3d.py
    echo "✓ src/unet_3d.py → src/core/unet_3d.py"
fi

# Preprocessing - shape_carving.py만 이동 (shape_carver.py는 수동 확인 필요)
if [ -f "src/shape_carving.py" ]; then
    git mv src/shape_carving.py src/preprocessing/shape_carving.py
    echo "✓ src/shape_carving.py → src/preprocessing/shape_carving.py"
fi

# Utils
if [ -f "src/config_utils.py" ]; then
    git mv src/config_utils.py src/utils/config_utils.py
    echo "✓ src/config_utils.py → src/utils/config_utils.py"
fi

if [ -f "src/tracking.py" ]; then
    git mv src/tracking.py src/utils/tracking.py
    echo "✓ src/tracking.py → src/utils/tracking.py"
fi

if [ -f "src/plots.py" ]; then
    git mv src/plots.py src/utils/plots.py
    echo "✓ src/plots.py → src/utils/plots.py"
fi

if [ -f "src/utils.py" ]; then
    git mv src/utils.py src/utils/general.py
    echo "✓ src/utils.py → src/utils/general.py"
fi

echo ""
echo "=========================================="
echo "Phase 9: __init__.py 파일 생성"
echo "=========================================="

# scripts/__init__.py
touch scripts/__init__.py
touch scripts/pipeline/__init__.py
touch scripts/training/__init__.py
touch scripts/analysis/__init__.py
touch scripts/features/__init__.py
touch scripts/utils/__init__.py

# src/__init__.py
touch src/core/__init__.py
touch src/preprocessing/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/analysis/__init__.py
# src/utils/__init__.py already exists

echo "✓ 모든 __init__.py 파일 생성 완료"

echo ""
echo "=========================================="
echo "리팩토리 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "1. Import 경로 업데이트 필요: python3 docs/update_imports.py"
echo "2. 문서 업데이트: README.md, docs/ANALYSIS_GUIDE.md 등"
echo "3. 테스트 실행: 각 스크립트 개별 실행 확인"
echo "4. Git commit: git commit -am \"Refactor: Reorganize project structure\""
echo ""
echo "⚠️  주의: shape_carver.py 파일은 수동으로 확인 및 처리 필요"
echo ""
