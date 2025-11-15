#!/bin/bash
# Monitor Pose Splatter Pipeline Progress

LOG_DIR="output/markerless_mouse_nerf/logs"
PROJECT_DIR="output/markerless_mouse_nerf"

echo "========================================="
echo "Pose Splatter Pipeline Monitor"
echo "========================================="
echo "Time: $(date)"
echo ""

# Check if processes are running
echo "=== Running Processes ==="
ps aux | grep -E "python3.*(calculate|write|train|evaluate)" | grep -v grep
echo ""

# Check log files
echo "=== Log Files Status ==="
if [ -d "$LOG_DIR" ]; then
    ls -lh "$LOG_DIR"/*.log 2>/dev/null || echo "No log files yet"
else
    echo "Log directory not created yet"
fi
echo ""

# Check output files
echo "=== Output Files ==="
echo "Project directory: $PROJECT_DIR"
ls -lh "$PROJECT_DIR"/*.npz "$PROJECT_DIR"/*.pt 2>/dev/null || echo "No checkpoint files yet"
echo ""

if [ -f "$PROJECT_DIR/images/images.h5" ]; then
    echo "Images HDF5: $(ls -lh $PROJECT_DIR/images/images.h5 | awk '{print $5}')"
fi

if [ -d "$PROJECT_DIR/images/images.zarr" ]; then
    echo "Images Zarr: $(du -sh $PROJECT_DIR/images/images.zarr | awk '{print $1}')"
fi
echo ""

# Check GPU usage
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Tail recent logs
echo "=== Recent Log Activity ==="
for log in "$LOG_DIR"/*.log; do
    if [ -f "$log" ]; then
        echo "--- $(basename $log) (last 5 lines) ---"
        tail -5 "$log"
        echo ""
    fi
done

echo "========================================="
echo "To continuously monitor, run:"
echo "  watch -n 10 ./monitor_pipeline.sh"
echo "========================================="
