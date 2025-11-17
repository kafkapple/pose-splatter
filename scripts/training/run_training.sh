#!/bin/bash
################################################################################
# Training Wrapper Script for Pose Splatter
#
# Usage:
#   bash scripts/training/run_training.sh <config.json> [--epochs N] [other args]
#
# Example:
#   bash scripts/training/run_training.sh configs/baseline/markerless_mouse_nerf.json --epochs 50
#
# Author: Pose Splatter Team
# Date: November 2024
################################################################################

set -e  # Exit on any error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No config file provided${NC}"
    echo "Usage: bash $0 <config.json> [--epochs N] [other args]"
    echo ""
    echo "Examples:"
    echo "  bash $0 configs/baseline/markerless_mouse_nerf.json --epochs 50"
    echo "  bash $0 configs/baseline/markerless_mouse_nerf.json --epochs 10 --lr 0.0001"
    exit 1
fi

CONFIG_FILE="$1"
shift  # Remove config from arguments

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Pose Splatter Training${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Config: ${GREEN}$CONFIG_FILE${NC}"
echo -e "Arguments: ${GREEN}$@${NC}"
echo ""

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}Warning: No conda environment activated${NC}"
    echo -e "${YELLOW}Activating 'splatter' environment...${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate splatter
    echo -e "${GREEN}✓ Activated splatter environment${NC}"
fi

echo -e "Conda environment: ${GREEN}$CONDA_DEFAULT_ENV${NC}"
echo ""

# Set environment variables
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo -e "${BLUE}Environment variables:${NC}"
echo -e "  PROJECT_ROOT: ${GREEN}$PROJECT_ROOT${NC}"
echo -e "  PYTHONPATH: ${GREEN}$PYTHONPATH${NC}"
echo -e "  PYTORCH_CUDA_ALLOC_CONF: ${GREEN}$PYTORCH_CUDA_ALLOC_CONF${NC}"
echo ""

# Check GPU
echo -e "${BLUE}Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo -e "  GPU: ${GREEN}$GPU_INFO${NC}"
else
    echo -e "  ${YELLOW}Warning: nvidia-smi not found${NC}"
fi
echo ""

# Get project name from config
PROJECT_NAME=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['project_directory'].rstrip('/').split('/')[-1])")
LOG_FILE="output/${PROJECT_NAME}/logs/training_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "output/${PROJECT_NAME}/logs"

echo -e "${BLUE}Log file: ${GREEN}$LOG_FILE${NC}"
echo ""

# Run training
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Starting Training...${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

python scripts/training/train_script.py "$CONFIG_FILE" "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo -e "${BLUE}================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    echo -e "Output directory: ${GREEN}output/${PROJECT_NAME}/${NC}"
    echo -e "Log file: ${GREEN}$LOG_FILE${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Evaluate model:"
    echo -e "     ${GREEN}python scripts/training/evaluate_model.py $CONFIG_FILE${NC}"
    echo ""
    echo -e "  2. Visualize results:"
    echo -e "     ${GREEN}bash scripts/visualization/run_all_visualization.sh${NC}"
else
    echo -e "${RED}Training failed with exit code $EXIT_CODE${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    echo -e "Check log file: ${YELLOW}$LOG_FILE${NC}"
fi

exit $EXIT_CODE
