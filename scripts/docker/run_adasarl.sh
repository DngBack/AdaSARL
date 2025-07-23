#!/bin/bash

# AdaSARL Docker Runner Script
# This script runs AdaSARL experiments in Docker

set -e

# Default values
DATASET="seq-cifar10"
BUFFER_SIZE="200"
EXPERIMENT_ID="adasarl_$(date +%Y%m%d_%H%M%S)"
GPU_ID="0"
EPOCHS="50"
BATCH_SIZE="32"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset DATASET       Dataset to use (default: seq-cifar10)"
    echo "  -b, --buffer-size SIZE      Buffer size (default: 200)"
    echo "  -e, --experiment-id ID      Experiment ID (default: auto-generated)"
    echo "  -g, --gpu GPU_ID            GPU ID to use (default: 0)"
    echo "  -n, --epochs EPOCHS         Number of epochs (default: 50)"
    echo "  -s, --batch-size SIZE       Batch size (default: 32)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Supported datasets:"
    echo "  seq-cifar10, seq-cifar100, seq-tinyimg"
    echo ""
    echo "Examples:"
    echo "  $0 -d seq-cifar10 -b 200"
    echo "  $0 -d seq-cifar100 -b 500 -e my_experiment"
    echo "  $0 -d seq-tinyimg -g 1 -n 100"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -b|--buffer-size)
            BUFFER_SIZE="$2"
            shift 2
            ;;
        -e|--experiment-id)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -n|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -s|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate dataset
case $DATASET in
    seq-cifar10|seq-cifar100|seq-tinyimg)
        ;;
    *)
        echo "Error: Unsupported dataset '$DATASET'"
        echo "Supported datasets: seq-cifar10, seq-cifar100, seq-tinyimg"
        exit 1
        ;;
esac

echo "=========================================="
echo "AdaSARL Docker Runner"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Buffer Size: $BUFFER_SIZE"
echo "Experiment ID: $EXPERIMENT_ID"
echo "GPU ID: $GPU_ID"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "Warning: NVIDIA Docker runtime not available. Running on CPU only."
    GPU_FLAG=""
else
    GPU_FLAG="--gpus all"
fi

# Create output directories
mkdir -p outputs logs checkpoints data

# Build Docker image if it doesn't exist
if [[ "$(docker images -q adasarl:latest 2> /dev/null)" == "" ]]; then
    echo "Building AdaSARL Docker image..."
    docker build -t adasarl:latest .
fi

# Run the experiment
echo "Starting AdaSARL experiment..."
docker run $GPU_FLAG \
    --rm \
    --name "adasarl_${EXPERIMENT_ID}" \
    -e CUDA_VISIBLE_DEVICES=$GPU_ID \
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
        --dataset $DATASET \
        --buffer_size $BUFFER_SIZE \
        --experiment_id $EXPERIMENT_ID \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --tensorboard \
        --csv_log

echo "=========================================="
echo "Experiment completed!"
echo "Results saved to: outputs/$EXPERIMENT_ID"
echo "Logs saved to: logs/$EXPERIMENT_ID"
echo "Checkpoints saved to: checkpoints/$EXPERIMENT_ID"
echo "=========================================="

# Start TensorBoard if requested
read -p "Start TensorBoard to view results? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting TensorBoard..."
    docker run -d \
        --rm \
        --name "adasarl_tensorboard_${EXPERIMENT_ID}" \
        -p 6006:6006 \
        -v "$(pwd)/outputs:/workspace/adasarl/outputs" \
        adasarl:latest \
        tensorboard --logdir=/workspace/adasarl/outputs --host=0.0.0.0 --port=6006
    
    echo "TensorBoard started at: http://localhost:6006"
    echo "Press Ctrl+C to stop TensorBoard"
    
    # Wait for user to stop TensorBoard
    trap 'echo "Stopping TensorBoard..."; docker stop adasarl_tensorboard_${EXPERIMENT_ID}' INT
    wait
fi 