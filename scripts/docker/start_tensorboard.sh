#!/bin/bash

# AdaSARL TensorBoard Starter Script
# This script starts TensorBoard for monitoring AdaSARL experiments

set -e

# Default values
PORT="6006"
LOG_DIR="./outputs"
CONTAINER_NAME="adasarl_tensorboard"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT            Port to run TensorBoard on (default: 6006)"
    echo "  -d, --log-dir DIR          Log directory to monitor (default: ./outputs)"
    echo "  -n, --name NAME            Container name (default: adasarl_tensorboard)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 -p 6007 -d ./my_experiments"
    echo "  $0 --port 6008 --name my_tensorboard"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory '$LOG_DIR' does not exist."
    echo "Please run an experiment first or specify a valid log directory."
    exit 1
fi

# Check if AdaSARL image exists
if [[ "$(docker images -q adasarl:latest 2> /dev/null)" == "" ]]; then
    echo "Error: AdaSARL Docker image not found."
    echo "Please build the image first: docker build -t adasarl:latest ."
    exit 1
fi

# Stop existing TensorBoard container if running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo "Stopping existing TensorBoard container..."
    docker stop "$CONTAINER_NAME" > /dev/null 2>&1
fi

# Remove existing container if it exists
if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
    docker rm "$CONTAINER_NAME" > /dev/null 2>&1
fi

# Start TensorBoard
echo "Starting TensorBoard..."
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo "Container name: $CONTAINER_NAME"

docker run -d \
    --rm \
    --name "$CONTAINER_NAME" \
    -p "$PORT:6006" \
    -v "$(realpath $LOG_DIR):/workspace/adasarl/outputs" \
    adasarl:latest \
    tensorboard --logdir=/workspace/adasarl/outputs --host=0.0.0.0 --port=6006

echo "=========================================="
echo "TensorBoard started successfully!"
echo "Access TensorBoard at: http://localhost:$PORT"
echo "Container name: $CONTAINER_NAME"
echo ""
echo "To stop TensorBoard:"
echo "  docker stop $CONTAINER_NAME"
echo ""
echo "To view logs:"
echo "  docker logs $CONTAINER_NAME"
echo "=========================================="

# Wait for user to stop TensorBoard
echo "Press Ctrl+C to stop TensorBoard"
trap 'echo "Stopping TensorBoard..."; docker stop $CONTAINER_NAME' INT
wait 