#!/bin/bash

# AdaSARL Quick Test Script
# This script runs a quick test to verify the Docker setup

set -e

echo "=========================================="
echo "AdaSARL Quick Test"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Create output directories
mkdir -p outputs logs checkpoints data

# Build Docker image if it doesn't exist
if [[ "$(docker images -q adasarl:latest 2> /dev/null)" == "" ]]; then
    echo "Building AdaSARL Docker image..."
    docker build -t adasarl:latest .
fi

# Test 1: AdaSARL Inference on CIFAR-10 (2 epochs)
echo "Test 1: AdaSARL Inference on CIFAR-10 (2 epochs)"
docker run --rm \
    --name "adasarl_quick_test_1" \
    -e PYTHONPATH=/workspace/adasarl \
    -v "$(pwd):/workspace/adasarl" \
    -v "$(pwd)/data:/workspace/adasarl/data" \
    -v "$(pwd)/outputs:/workspace/adasarl/outputs" \
    -v "$(pwd)/logs:/workspace/adasarl/logs" \
    -v "$(pwd)/checkpoints:/workspace/adasarl/checkpoints" \
    -w /workspace/adasarl \
    adasarl:latest \
    python main.py \
        --model adasarl_inference \
        --dataset seq-cifar10 \
        --buffer_size 50 \
        --experiment_id quick_test_inference \
        --epochs 2 \
        --batch_size 16 \
        --csv_log

echo "Test 1 completed!"

# Test 2: AdaSARL on CIFAR-10 (2 epochs)
echo "Test 2: AdaSARL on CIFAR-10 (2 epochs)"
docker run --rm \
    --name "adasarl_quick_test_2" \
    -e PYTHONPATH=/workspace/adasarl \
    -v "$(pwd):/workspace/adasarl" \
    -v "$(pwd)/data:/workspace/adasarl/data" \
    -v "$(pwd)/outputs:/workspace/adasarl/outputs" \
    -v "$(pwd)/logs:/workspace/adasarl/logs" \
    -v "$(pwd)/checkpoints:/workspace/adasarl/checkpoints" \
    -w /workspace/adasarl \
    adasarl:latest \
    python main.py \
        --model adasarl \
        --dataset seq-cifar10 \
        --buffer_size 50 \
        --experiment_id quick_test_adasarl \
        --epochs 2 \
        --batch_size 16 \
        --csv_log

echo "Test 2 completed!"

# Test 3: Check if models can be imported
echo "Test 3: Model import test"
docker run --rm \
    --name "adasarl_import_test" \
    -e PYTHONPATH=/workspace/adasarl \
    -v "$(pwd):/workspace/adasarl" \
    -w /workspace/adasarl \
    adasarl:latest \
    python -c "
import sys
sys.path.append('/workspace/adasarl')
try:
    from models.adasarl_inference import SARLEnhancedGeluBalancedInference
    from models.adasarl import SARLEnhancedGeluBalanced
    print('✓ AdaSARL Inference model imported successfully')
    print('✓ AdaSARL model imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo "Results saved to: outputs/quick_test_*"
echo "Logs saved to: logs/quick_test_*"
echo ""
echo "Next steps:"
echo "1. Run full experiment: ./scripts/docker/run_adasarl_inference.sh"
echo "2. Start TensorBoard: docker run -d -p 6006:6006 -v \$(pwd)/outputs:/workspace/adasarl/outputs adasarl:latest tensorboard --logdir=/workspace/adasarl/outputs --host=0.0.0.0 --port=6006"
echo "3. View results at: http://localhost:6006" 