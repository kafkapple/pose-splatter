#!/bin/bash

# Wrapper for export_temporal_sequence_rerun.py with proper environment setup

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Activate conda environment
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate splatter 2>/dev/null || true

# Run script
python scripts/visualization/export_temporal_sequence_rerun.py "$@"
