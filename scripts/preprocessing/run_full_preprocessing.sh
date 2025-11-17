#!/bin/bash
################################################################################
# Full Preprocessing Pipeline for Pose Splatter
#
# This script runs all preprocessing steps in sequence:
#   1. Estimate up direction (auto mode)
#   2. Calculate center and rotation
#   3. Calculate crop indices
#   4. Write images to HDF5
#   5. Copy to ZARR for training
#
# Usage:
#   bash scripts/preprocessing/run_full_preprocessing.sh <config_file.json>
#
# Example:
#   bash scripts/preprocessing/run_full_preprocessing.sh configs/baseline/markerless_mouse_nerf.json
#
# Author: Pose Splatter Team
# Date: November 2024
################################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No config file provided${NC}"
    echo "Usage: bash $0 <config_file.json>"
    echo "Example: bash $0 configs/baseline/markerless_mouse_nerf.json"
    exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Get project name from config file
PROJECT_NAME=$(basename "$CONFIG_FILE" .json)
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Preprocessing Pipeline${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Config: ${GREEN}$CONFIG_FILE${NC}"
echo -e "Project: ${GREEN}$PROJECT_NAME${NC}"
echo ""

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}Warning: No conda environment activated${NC}"
    echo -e "${YELLOW}Using: conda run -n splatter${NC}"
    CONDA_PREFIX="conda run -n splatter"
else
    echo -e "Conda environment: ${GREEN}$CONDA_DEFAULT_ENV${NC}"
    CONDA_PREFIX=""
fi
echo ""

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
if [ -z "$CONDA_PREFIX" ]; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="conda run -n splatter python"
fi

$PYTHON_CMD -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: opencv-python not installed${NC}"
    echo -e "${YELLOW}Installing opencv-python...${NC}"
    if [ -z "$CONDA_PREFIX" ]; then
        pip install opencv-python tqdm
    else
        conda run -n splatter pip install opencv-python tqdm
    fi
    echo -e "${GREEN}✓ opencv-python installed${NC}"
fi
echo ""

# Set environment variables
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo -e "${BLUE}Environment variables set:${NC}"
echo -e "  PYTHONPATH: $PYTHONPATH"
echo -e "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Create output directories
OUTPUT_DIR="output/${PROJECT_NAME}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
echo -e "Output directory: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Logs directory: ${GREEN}$LOG_DIR${NC}"
echo ""

# Parse config to get data directory (handle both relative and absolute paths)
DATA_DIR=$(python -c "import json, os; d=json.load(open('$CONFIG_FILE'))['data_directory']; print(d if os.path.isabs(d) else os.path.join(os.getcwd(), d))")
echo -e "Data directory: ${GREEN}$DATA_DIR${NC}"
echo ""

# Step 0: Copy camera parameters to output directory
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Step 0: Setup${NC}"
echo -e "${BLUE}================================${NC}"

CAMERA_SRC="$DATA_DIR/camera_params.h5"
CAMERA_DST="$OUTPUT_DIR/camera_params.h5"

# Check if camera_params.h5 exists, if not try to convert from new_cam.pkl
if [ ! -f "$CAMERA_SRC" ]; then
    CAMERA_PKL="$DATA_DIR/new_cam.pkl"
    if [ -f "$CAMERA_PKL" ]; then
        echo -e "${YELLOW}camera_params.h5 not found, converting from new_cam.pkl...${NC}"
        if [ -z "$CONDA_PREFIX" ]; then
            python scripts/preprocessing/convert_camera_params.py "$CAMERA_PKL" "$CAMERA_SRC"
        else
            $CONDA_PREFIX python scripts/preprocessing/convert_camera_params.py "$CAMERA_PKL" "$CAMERA_SRC"
        fi
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Converted new_cam.pkl to camera_params.h5${NC}"
        else
            echo -e "${RED}Error: Failed to convert camera parameters${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: Neither camera_params.h5 nor new_cam.pkl found in: $DATA_DIR${NC}"
        echo -e "${YELLOW}Expected files:${NC}"
        echo -e "  - $CAMERA_SRC (preferred)"
        echo -e "  - $CAMERA_PKL (will be converted)"
        exit 1
    fi
fi

cp "$CAMERA_SRC" "$CAMERA_DST"
echo -e "${GREEN}✓ Copied camera parameters${NC}"
echo -e "  From: $CAMERA_SRC"
echo -e "  To: $CAMERA_DST"
echo ""

# Function to run a step with logging
run_step() {
    local step_num=$1
    local step_name=$2
    local script_name=$3
    local log_file="$LOG_DIR/step${step_num}_${step_name}.log"

    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Step $step_num: $step_name${NC}"
    echo -e "${BLUE}================================${NC}"
    echo -e "Script: ${GREEN}scripts/preprocessing/$script_name${NC}"
    echo -e "Log: ${GREEN}$log_file${NC}"
    echo ""

    # Run the script (use absolute path to ensure correct file)
    local script_path="$(pwd)/scripts/preprocessing/$script_name"

    if [ ! -f "$script_path" ]; then
        echo -e "${RED}Error: Script not found: $script_path${NC}"
        exit 1
    fi

    if [ -z "$CONDA_PREFIX" ]; then
        python "$script_path" "$CONFIG_FILE" 2>&1 | tee "$log_file"
    else
        $CONDA_PREFIX python "$script_path" "$CONFIG_FILE" 2>&1 | tee "$log_file"
    fi

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Step $step_num completed successfully${NC}"
    else
        echo -e "${RED}✗ Step $step_num failed with exit code $exit_code${NC}"
        echo -e "${RED}Check log: $log_file${NC}"
        exit $exit_code
    fi
    echo ""
}

# Step 1: Estimate up direction (auto mode)
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Step 1: Estimate up direction${NC}"
echo -e "${BLUE}================================${NC}"

UP_OUTPUT="$OUTPUT_DIR/vertical_lines.npz"
LOG_FILE="$LOG_DIR/step1_up_direction.log"

echo -e "Input: ${GREEN}$CAMERA_DST${NC}"
echo -e "Output: ${GREEN}$UP_OUTPUT${NC}"
echo -e "Log: ${GREEN}$LOG_FILE${NC}"
echo ""

if [ -z "$CONDA_PREFIX" ]; then
    python scripts/preprocessing/auto_estimate_up.py "$CAMERA_DST" "$UP_OUTPUT" 2>&1 | tee "$LOG_FILE"
else
    $CONDA_PREFIX python scripts/preprocessing/auto_estimate_up.py "$CAMERA_DST" "$UP_OUTPUT" 2>&1 | tee "$LOG_FILE"
fi

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✓ Step 1 completed successfully${NC}"
else
    echo -e "${RED}✗ Step 1 failed with exit code $exit_code${NC}"
    exit $exit_code
fi
echo ""

# Step 2: Calculate center and rotation
run_step 2 "center_rotation" "calculate_center_rotation.py"

# Step 3: Calculate crop indices
run_step 3 "crop_indices" "calculate_crop_indices.py"

# Step 4: Write images to HDF5
run_step 4 "write_images" "write_images.py"

# Step 5: Copy to ZARR
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Step 5: Copy to ZARR${NC}"
echo -e "${BLUE}================================${NC}"

H5_FILE="output/${PROJECT_NAME}/images/images.h5"
ZARR_FILE="output/${PROJECT_NAME}/images/images.zarr"
LOG_FILE="$LOG_DIR/step5_copy_to_zarr.log"

if [ ! -f "$H5_FILE" ]; then
    echo -e "${RED}Error: HDF5 file not found: $H5_FILE${NC}"
    exit 1
fi

echo -e "Input: ${GREEN}$H5_FILE${NC}"
echo -e "Output: ${GREEN}$ZARR_FILE${NC}"
echo -e "Log: ${GREEN}$LOG_FILE${NC}"
echo ""

if [ -z "$CONDA_PREFIX" ]; then
    python scripts/preprocessing/copy_to_zarr.py "$H5_FILE" "$ZARR_FILE" 2>&1 | tee "$LOG_FILE"
else
    $CONDA_PREFIX python scripts/preprocessing/copy_to_zarr.py "$H5_FILE" "$ZARR_FILE" 2>&1 | tee "$LOG_FILE"
fi

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✓ Step 5 completed successfully${NC}"
else
    echo -e "${RED}✗ Step 5 failed with exit code $exit_code${NC}"
    echo -e "${RED}Check log: $LOG_FILE${NC}"
    exit $exit_code
fi
echo ""

# Summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Preprocessing Complete!${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo -e "${GREEN}✓ All preprocessing steps completed successfully${NC}"
echo ""
echo -e "Output directory: ${GREEN}output/${PROJECT_NAME}/${NC}"
echo -e "  - Images (HDF5): ${GREEN}$H5_FILE${NC}"
echo -e "  - Images (ZARR): ${GREEN}$ZARR_FILE${NC}"
echo -e "  - Logs: ${GREEN}$LOG_DIR/${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Verify dataset:"
echo -e "     ${GREEN}python scripts/utils/verify_datasets.py${NC}"
echo ""
echo -e "  2. Train model:"
echo -e "     ${GREEN}conda run -n splatter python scripts/training/train_script.py $CONFIG_FILE --epochs 50${NC}"
echo ""
echo -e "  3. Monitor training:"
echo -e "     ${GREEN}tail -f output/${PROJECT_NAME}/logs/step6_training.log${NC}"
echo ""
