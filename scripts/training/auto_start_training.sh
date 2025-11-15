#!/bin/bash

# Auto-start Pose Splatter training after mouse-super-resolution completes
# Created: 2025-11-09 17:12

echo "=== Pose Splatter Auto-Training Script ==="
echo "Waiting for mouse-super-resolution to complete..."
echo "Expected completion: ~17:26 (in 15 minutes)"
echo ""

# Monitor mouse-super-resolution process
MOUSE_SR_PID=1116776

# Wait for process to complete
while kill -0 $MOUSE_SR_PID 2>/dev/null; do
    echo "[$(date '+%H:%M:%S')] mouse-super-resolution still running (PID: $MOUSE_SR_PID)..."
    sleep 60  # Check every minute
done

echo ""
echo "[$(date '+%H:%M:%S')] ✅ mouse-super-resolution completed!"
echo "Waiting 10 seconds to ensure GPU is free..."
sleep 10

# Verify GPU is available
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo ""

# Start Pose Splatter training
echo "=== Starting Pose Splatter Training ==="
echo "Config: configs/markerless_mouse_nerf.json"
echo "Epochs: 50"
echo "Output: output/markerless_mouse_nerf/"
echo ""

cd /home/joon/dev/pose-splatter

# Start training
python3 train_script.py configs/markerless_mouse_nerf.json --epochs 50 2>&1 | tee output/markerless_mouse_nerf/logs/step6_training.log

echo ""
echo "[$(date '+%H:%M:%S')] ✅ Training completed!"
echo "Check results at: output/markerless_mouse_nerf/"
