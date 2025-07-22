# AdaSARL Docker Setup

This document provides comprehensive instructions for running AdaSARL experiments using Docker.

## 🐳 Quick Start

### Prerequisites

1. **Docker**: Install Docker Desktop or Docker Engine
2. **NVIDIA Docker** (optional): For GPU acceleration
3. **Git**: To clone the repository

### Installation

```bash
# Clone the repository
git clone https://github.com/DngBack/AdaSARL.git
cd AdaSARL

# Make scripts executable
chmod +x scripts/docker/*.sh

# Build the Docker image
docker build -t adasarl:latest .
```

## 🚀 Running Experiments

### Quick Test (Recommended First Step)

```bash
# Run a quick test to verify everything works
./scripts/docker/quick_test.sh
```

This will:

- Test AdaSARL Inference on CIFAR-10 (2 epochs)
- Test AdaSARL on CIFAR-10 (2 epochs)
- Verify model imports work correctly

### Run AdaSARL Inference (Recommended)

```bash
# Basic run on CIFAR-10
./scripts/docker/run_adasarl_inference.sh

# Custom parameters
./scripts/docker/run_adasarl_inference.sh \
    -d seq-cifar100 \
    -b 500 \
    -e my_experiment \
    -g 0 \
    -n 100 \
    -s 64
```

### Run AdaSARL (Training + Inference)

```bash
# Basic run on CIFAR-10
./scripts/docker/run_adasarl.sh

# Custom parameters
./scripts/docker/run_adasarl.sh \
    -d seq-cifar100 \
    -b 500 \
    -e my_experiment \
    -g 0 \
    -n 100 \
    -s 64
```

## 📊 Monitoring with TensorBoard

### Start TensorBoard

```bash
# Start TensorBoard to monitor experiments
./scripts/docker/start_tensorboard.sh

# Custom port and log directory
./scripts/docker/start_tensorboard.sh \
    -p 6007 \
    -d ./my_experiments \
    -n my_tensorboard
```

### Access TensorBoard

Open your browser and go to: `http://localhost:6006`

## 🛠️ Manual Docker Commands

### Build Image

```bash
docker build -t adasarl:latest .
```

### Run Container

```bash
# Interactive shell
docker run -it --rm \
    --gpus all \
    -v $(pwd):/workspace/adasarl \
    -v $(pwd)/data:/workspace/adasarl/data \
    -v $(pwd)/outputs:/workspace/adasarl/outputs \
    -v $(pwd)/logs:/workspace/adasarl/logs \
    -v $(pwd)/checkpoints:/workspace/adasarl/checkpoints \
    -w /workspace/adasarl \
    adasarl:latest bash

# Run experiment directly
docker run --rm \
    --gpus all \
    -v $(pwd):/workspace/adasarl \
    -v $(pwd)/data:/workspace/adasarl/data \
    -v $(pwd)/outputs:/workspace/adasarl/outputs \
    -v $(pwd)/logs:/workspace/adasarl/logs \
    -v $(pwd)/checkpoints:/workspace/adasarl/checkpoints \
    -w /workspace/adasarl \
    adasarl:latest \
    python main.py \
        --model adasarl_inference \
        --dataset seq-cifar10 \
        --buffer_size 200 \
        --experiment_id docker_test \
        --epochs 50
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# Run experiment in the container
docker-compose exec adasarl python main.py \
    --model adasarl_inference \
    --dataset seq-cifar10 \
    --buffer_size 200 \
    --experiment_id compose_test \
    --epochs 50

# Stop services
docker-compose down
```

## 📁 Directory Structure

```
AdaSARL/
├── Dockerfile                    # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── .dockerignore               # Files to exclude from Docker build
├── scripts/docker/             # Docker utility scripts
│   ├── run_adasarl_inference.sh # Run AdaSARL Inference
│   ├── run_adasarl.sh          # Run AdaSARL
│   ├── quick_test.sh           # Quick test script
│   └── start_tensorboard.sh    # Start TensorBoard
├── outputs/                    # Experiment results (mounted)
├── logs/                      # Log files (mounted)
├── checkpoints/               # Model checkpoints (mounted)
└── data/                      # Dataset files (mounted)
```

## ⚙️ Configuration Options

### Script Parameters

| Parameter             | Description        | Default        | Options                                      |
| --------------------- | ------------------ | -------------- | -------------------------------------------- |
| `-d, --dataset`       | Dataset to use     | `seq-cifar10`  | `seq-cifar10`, `seq-cifar100`, `seq-tinyimg` |
| `-b, --buffer-size`   | Memory buffer size | `200`          | Any positive integer                         |
| `-e, --experiment-id` | Experiment ID      | Auto-generated | Any string                                   |
| `-g, --gpu`           | GPU ID to use      | `0`            | `0`, `1`, `2`, etc.                          |
| `-n, --epochs`        | Number of epochs   | `50`           | Any positive integer                         |
| `-s, --batch-size`    | Batch size         | `32`           | Any positive integer                         |

### Environment Variables

| Variable               | Description              | Default              |
| ---------------------- | ------------------------ | -------------------- |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use       | `0`                  |
| `PYTHONPATH`           | Python path              | `/workspace/adasarl` |
| `PYTHONUNBUFFERED`     | Unbuffered Python output | `1`                  |

## 🔧 Troubleshooting

### Common Issues

1. **Docker not running**

   ```bash
   # Start Docker Desktop or Docker service
   sudo systemctl start docker  # Linux
   # Or start Docker Desktop on Windows/Mac
   ```

2. **NVIDIA Docker not available**

   ```bash
   # Install NVIDIA Docker runtime
   # Follow instructions at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

3. **Permission denied**

   ```bash
   # Make scripts executable
   chmod +x scripts/docker/*.sh
   ```

4. **Out of memory**

   ```bash
   # Reduce batch size or buffer size
   ./scripts/docker/run_adasarl_inference.sh -s 16 -b 100
   ```

5. **Port already in use**
   ```bash
   # Use different port for TensorBoard
   ./scripts/docker/start_tensorboard.sh -p 6007
   ```

### Debug Commands

```bash
# Check Docker status
docker info

# Check GPU availability
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi

# Check container logs
docker logs <container_name>

# Enter running container
docker exec -it <container_name> bash

# Check image details
docker inspect adasarl:latest
```

## 📈 Performance Tips

### GPU Optimization

1. **Use GPU**: Ensure NVIDIA Docker runtime is installed
2. **Memory**: Monitor GPU memory usage with `nvidia-smi`
3. **Batch Size**: Adjust based on available GPU memory

### Storage Optimization

1. **Volumes**: Use Docker volumes for persistent data
2. **Cleanup**: Regularly clean up old containers and images
   ```bash
   docker system prune -a
   ```

### Network Optimization

1. **Data Location**: Keep datasets close to the container
2. **Bandwidth**: Use local data sources when possible

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: AdaSARL Docker Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t adasarl:latest .
      - name: Run quick test
        run: ./scripts/docker/quick_test.sh
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch Docker](https://hub.docker.com/r/pytorch/pytorch)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)

## 🤝 Contributing

When contributing to the Docker setup:

1. Test your changes with the quick test script
2. Update this documentation if needed
3. Ensure backward compatibility
4. Add appropriate error handling

## 📞 Support

For Docker-related issues:

1. Check the troubleshooting section
2. Verify your Docker installation
3. Check the logs with `docker logs <container_name>`
4. Open an issue with detailed error information
