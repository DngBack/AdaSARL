#!/bin/bash

# Docker wrapper script for running seq-cifar100 grid search
# Usage: ./run_seq_cifar100.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AdaSARL Sequential CIFAR-100 Grid Search in Docker${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if nvidia-docker is available
if ! command -v nvidia-docker &> /dev/null && ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not found. GPU acceleration may not work.${NC}"
fi

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t adasarl:latest .

# Create necessary directories on host
mkdir -p ./outputs/experiments/adasarl
mkdir -p ./logs
mkdir -p ./checkpoints
mkdir -p ./data

# Run the grid search in Docker
echo -e "${GREEN}Running CIFAR-100 grid search experiments...${NC}"
docker run --rm \
    --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH=/workspace/adasarl \
    -v $(pwd):/workspace/adasarl \
    -v $(pwd)/outputs:/workspace/adasarl/outputs \
    -v $(pwd)/logs:/workspace/adasarl/logs \
    -v $(pwd)/checkpoints:/workspace/adasarl/checkpoints \
    -v $(pwd)/data:/workspace/adasarl/data \
    -w /workspace/adasarl \
    adasarl:latest \
    python scripts/adasarl/seq-cifar100.py

echo -e "${GREEN}CIFAR-100 grid search completed! Check outputs/experiments/adasarl for results.${NC}"
