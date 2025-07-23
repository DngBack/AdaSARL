#!/bin/bash

# Master script to run all AdaSARL experiments
# Usage: ./run_all_experiments.sh [--adasarl-only|--inference-only]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default: run both
RUN_ADASARL=true
RUN_INFERENCE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --adasarl-only)
            RUN_ADASARL=true
            RUN_INFERENCE=false
            shift
            ;;
        --inference-only)
            RUN_ADASARL=false
            RUN_INFERENCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--adasarl-only|--inference-only]"
            echo ""
            echo "Options:"
            echo "  --adasarl-only     Run only AdaSARL experiments"
            echo "  --inference-only   Run only AdaSARL Inference experiments"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "By default, runs both AdaSARL and AdaSARL Inference experiments"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  AdaSARL Complete Experiment Suite${NC}"
echo -e "${BLUE}========================================${NC}"

if [ "$RUN_ADASARL" = true ]; then
    echo -e "${GREEN}Running AdaSARL Experiments...${NC}"
    echo -e "${YELLOW}This will run 3 datasets × 3 seeds × multiple hyperparameters${NC}"
    echo -e "${YELLOW}Expected time: Several hours to days depending on hardware${NC}"
    echo ""
    
    echo -e "${GREEN}1/3: Running CIFAR-10 experiments...${NC}"
    ./scripts/docker/run_seq_cifar10.sh
    
    echo -e "${GREEN}2/3: Running CIFAR-100 experiments...${NC}"
    ./scripts/docker/run_seq_cifar100.sh
    
    echo -e "${GREEN}3/3: Running TinyImageNet experiments...${NC}"
    ./scripts/docker/run_seq_tinyimg.sh
    
    echo -e "${GREEN}✅ AdaSARL experiments completed!${NC}"
fi

if [ "$RUN_INFERENCE" = true ]; then
    echo -e "${GREEN}Running AdaSARL Inference Experiments...${NC}"
    echo -e "${YELLOW}This will run 3 datasets × 3 seeds × multiple hyperparameters${NC}"
    echo -e "${YELLOW}Expected time: Several hours to days depending on hardware${NC}"
    echo ""
    
    echo -e "${GREEN}1/3: Running CIFAR-10 inference experiments...${NC}"
    ./scripts/docker/run_inference_seq_cifar10.sh
    
    echo -e "${GREEN}2/3: Running CIFAR-100 inference experiments...${NC}"
    ./scripts/docker/run_inference_seq_cifar100.sh
    
    echo -e "${GREEN}3/3: Running TinyImageNet inference experiments...${NC}"
    ./scripts/docker/run_inference_seq_tinyimg.sh
    
    echo -e "${GREEN}✅ AdaSARL Inference experiments completed!${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🎉 All experiments completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Results can be found in:${NC}"
if [ "$RUN_ADASARL" = true ]; then
    echo -e "  📁 outputs/experiments/adasarl/"
fi
if [ "$RUN_INFERENCE" = true ]; then
    echo -e "  📁 outputs/experiments/adasarl_inference/"
fi
echo ""
echo -e "${YELLOW}To monitor progress:${NC}"
echo -e "  📊 TensorBoard: docker run --rm -p 6006:6006 -v \$(pwd)/outputs:/workspace/outputs adasarl:latest tensorboard --logdir=/workspace/outputs --host=0.0.0.0"
echo -e "  📋 CSV logs: Check the CSV files in the output directories"
