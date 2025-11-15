#!/bin/bash
# 2D vs 3D Gaussian Splatting 비교 실험 자동화 스크립트
# Usage: bash scripts/run_2d_3d_comparison.sh [--phase1|--phase2]

set -e  # Exit on error

PROJECT_ROOT="/home/joon/dev/pose-splatter"
cd $PROJECT_ROOT

# Conda 환경 설정
CONDA_ENV="splatter"

# 환경 변수
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 날짜 태그
DATE_TAG=$(date +%Y%m%d_%H%M)

# 색상 코드
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 헬퍼 함수
print_header() {
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# GPU 메모리 확인
check_gpu() {
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
}

# 환경 검증
check_environment() {
    print_header "Environment Check"

    # Conda 환경 확인
    echo "Conda Environment: $CONDA_ENV"

    # Python
    conda run -n $CONDA_ENV python -c "import sys; print(f'Python: {sys.version.split()[0]}')"

    # PyTorch
    conda run -n $CONDA_ENV python -c "import torch; print(f'PyTorch: {torch.__version__}')"

    # CUDA
    if conda run -n $CONDA_ENV python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        print_success "CUDA available"
    else
        print_error "CUDA not available"
        exit 1
    fi

    # Numpy
    conda run -n $CONDA_ENV python -c "import numpy; print(f'Numpy: {numpy.__version__}')"

    # GPU
    check_gpu
}

# Config 검증
validate_config() {
    local config_file=$1
    print_header "Validating Config: $config_file"

    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        exit 1
    fi

    # 주요 파라미터 출력
    echo "Key Parameters:"
    conda run -n $CONDA_ENV python -c "
import json
with open('$config_file') as f:
    config = json.load(f)
print(f\"  grid_size: {config.get('grid_size', 'N/A')}\")
print(f\"  max_frames: {config.get('max_frames', 'N/A')}\")
print(f\"  gaussian_mode: {config.get('gaussian_mode', 'N/A')}\")
print(f\"  project_directory: {config.get('project_directory', 'N/A')}\")
"
    echo ""
}

# 학습 실행 및 검증
run_training() {
    local config_file=$1
    local epochs=$2
    local log_file=$3
    local mode=$(basename $config_file | grep -oP '(2d|3d)')

    print_header "${mode^^} Training: $epochs epochs"

    # Config 검증
    validate_config $config_file

    # 학습 실행
    echo "Starting training..."
    echo "Config: $config_file"
    echo "Epochs: $epochs"
    echo "Log: $log_file"
    echo ""

    # 시작 시간 기록
    start_time=$(date +%s)

    # 학습 실행
    conda run -n $CONDA_ENV python scripts/training/train_script.py $config_file --epochs $epochs > $log_file 2>&1

    # 종료 시간 기록
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # 결과 검증
    if [ $? -eq 0 ]; then
        print_success "${mode^^} Training Complete (${duration}s)"

        # 마지막 3개 epoch loss 출력
        echo ""
        echo "Final losses:"
        grep "epoch loss" $log_file | tail -3
        echo ""

        # Checkpoint 확인
        project_dir=$(conda run -n $CONDA_ENV python -c "import json; f = open('$config_file'); config = json.load(f); f.close(); print(config['project_directory'])")
        if [ -f "${project_dir}/checkpoint.pt" ]; then
            print_success "Checkpoint saved: ${project_dir}/checkpoint.pt"
            ls -lh "${project_dir}/checkpoint.pt"
        else
            print_warning "Checkpoint not found"
        fi

        echo ""
        return 0
    else
        print_error "${mode^^} Training Failed"
        echo ""
        echo "Last 30 lines of log:"
        tail -30 $log_file
        echo ""
        echo "Full log: $log_file"
        return 1
    fi
}

# Phase 1: Debug Mode
phase1_debug() {
    print_header "Phase 1: Debug Mode (10 epochs)"

    echo "Goal: Validate configs and prevent OOM"
    echo "Expected time: 40-60 minutes (20-30 min each)"
    echo ""

    # 환경 확인
    check_environment

    # 2D Debug
    LOG_2D_DEBUG="output/logs/2d_debug_${DATE_TAG}.log"
    if run_training "configs/debug/2d_3d_comparison_2d_debug.json" 10 "$LOG_2D_DEBUG"; then
        print_success "Phase 1.1 (2D Debug) PASSED"
    else
        print_error "Phase 1.1 (2D Debug) FAILED"
        exit 1
    fi

    # GPU 메모리 체크
    check_gpu

    # 3D Debug
    LOG_3D_DEBUG="output/logs/3d_debug_${DATE_TAG}.log"
    if run_training "configs/debug/2d_3d_comparison_3d_debug.json" 10 "$LOG_3D_DEBUG"; then
        print_success "Phase 1.2 (3D Debug) PASSED"
    else
        print_error "Phase 1.2 (3D Debug) FAILED"
        exit 1
    fi

    # Phase 1 요약
    print_header "Phase 1 Complete!"
    echo "2D Debug Log: $LOG_2D_DEBUG"
    echo "3D Debug Log: $LOG_3D_DEBUG"
    echo ""
    print_success "All configs validated successfully"
    echo ""
    echo "Next Steps:"
    echo "  1. Review debug results"
    echo "  2. Create Phase 2 configs (2d_short, 3d_short)"
    echo "  3. Run Phase 2: bash scripts/run_2d_3d_comparison.sh --phase2"
    echo ""
}

# Phase 2: Short Training
phase2_short() {
    print_header "Phase 2: Short Training (50 epochs)"

    echo "Goal: Compare 2D vs 3D performance"
    echo "Expected time: 4-6 hours total"
    echo ""

    # 환경 확인
    check_environment

    # Config 파일 확인
    if [ ! -f "configs/2d_3d_comparison_2d_short.json" ]; then
        print_error "Phase 2 config not found: configs/2d_3d_comparison_2d_short.json"
        echo "Please create Phase 2 configs first"
        exit 1
    fi

    # 2D Short Training
    LOG_2D_SHORT="output/2d_short_${DATE_TAG}.log"
    if run_training "configs/2d_3d_comparison_2d_short.json" 50 "$LOG_2D_SHORT"; then
        print_success "Phase 2.1 (2D Short) PASSED"
    else
        print_error "Phase 2.1 (2D Short) FAILED"
        exit 1
    fi

    # GPU 메모리 체크
    check_gpu

    # 3D Short Training
    LOG_3D_SHORT="output/3d_short_${DATE_TAG}.log"
    if run_training "configs/2d_3d_comparison_3d_short.json" 50 "$LOG_3D_SHORT"; then
        print_success "Phase 2.2 (3D Short) PASSED"
    else
        print_error "Phase 2.2 (3D Short) FAILED"
        exit 1
    fi

    # Phase 2 요약
    print_header "Phase 2 Complete!"
    echo "2D Short Log: $LOG_2D_SHORT"
    echo "3D Short Log: $LOG_3D_SHORT"
    echo ""
    print_success "Both experiments completed successfully"
    echo ""
    echo "Next Steps:"
    echo "  1. Run analysis: python3 scripts/analyze_results.py"
    echo "  2. Review comparison results"
    echo "  3. Decide on Phase 3 (full training) if needed"
    echo ""
}

# Main
main() {
    print_header "2D vs 3D Gaussian Splatting Comparison"
    echo "Date: $(date)"
    echo "Project: $PROJECT_ROOT"
    echo ""

    # 인자 파싱
    case "$1" in
        --phase1)
            phase1_debug
            ;;
        --phase2)
            phase2_short
            ;;
        *)
            echo "Usage: bash scripts/run_2d_3d_comparison.sh [--phase1|--phase2]"
            echo ""
            echo "Options:"
            echo "  --phase1    Run Phase 1: Debug Mode (10 epochs each, ~1 hour)"
            echo "  --phase2    Run Phase 2: Short Training (50 epochs each, ~5 hours)"
            echo ""
            echo "Example:"
            echo "  bash scripts/run_2d_3d_comparison.sh --phase1"
            echo ""
            exit 1
            ;;
    esac
}

# 실행
main "$@"
