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

### 🎯 Complete Experiment Suite

#### Run All Experiments (Recommended for Full Results)

```bash
# Run all AdaSARL and AdaSARL Inference experiments
./scripts/docker/run_all_experiments.sh

# Run only AdaSARL experiments
./scripts/docker/run_all_experiments.sh --adasarl-only

# Run only AdaSARL Inference experiments
./scripts/docker/run_all_experiments.sh --inference-only
```

**⚠️ Time Requirements:**
- **Total experiments**: ~144 experiments (6 experiment types × 3 datasets × 3 seeds × ~2-4 hyperparameter combinations each)
- **Expected time**: 2-7 days depending on hardware
- **GPU memory**: Requires ~8GB+ GPU memory

#### Individual Experiment Types

#### � Output Structure

After running experiments, your output directory will be organized as follows:

```
outputs/
├── experiments/
│   ├── adasarl/                           # AdaSARL experiment results
│   │   ├── adasarl-cifar10-200-param-*-s-1/
│   │   ├── adasarl-cifar10-200-param-*-s-3/
│   │   ├── adasarl-cifar10-200-param-*-s-5/
│   │   ├── adasarl-cifar10-500-param-*-s-*/
│   │   ├── adasarl-cifar100-*-s-*/
│   │   └── adasarl-tinyimg-*-s-*/
│   └── adasarl_inference/                 # AdaSARL Inference experiment results
│       ├── adasarl_inference-cifar10-200-param-*-s-1/
│       ├── adasarl_inference-cifar10-200-param-*-s-3/
│       ├── adasarl_inference-cifar10-200-param-*-s-5/
│       └── ... (similar structure)
├── logs/                                  # Training logs
└── tensorboard/                          # TensorBoard logs
```

Each experiment directory contains:
- `results.csv`: Detailed metrics for each task
- `model_final.pth`: Final trained model (if save_model=1)
- `config.json`: Experiment configuration
- TensorBoard logs for visualization

### 🔬 Experiment Details

#### Hyperparameter Grid Search

**AdaSARL Experiments** test combinations of:
- `alpha`: [0.5] (buffer_size=200), [0.2] (buffer_size=500)
- `beta`: [1.0]
- `op_weight`: [0.5, 0.7]
- `sm_weight`: [0.01, 0.05]
- `num_feats`: [512, 1024]
- `balance_weight`: [1.0, 2.0]
- Buffer sizes: [200, 500]
- Seeds: [1, 3, 5]

**AdaSARL Inference Experiments** use similar parameters with:
- Inference-only prototype guidance
- Conservative guidance weights
- High momentum updates

#### Expected Results

**CIFAR-10**: Higher accuracy due to simpler 10-class classification
**CIFAR-100**: More challenging 100-class classification
**TinyImageNet**: Most complex with 200 classes and higher resolution

**Metrics tracked**:
- Accuracy after each task
- Average accuracy across all tasks
- Forgetting measures
- Training time per task

## 📊 Advanced Usage

These experiments test the full AdaSARL model with GELU activation and balanced sampling:

```bash
# CIFAR-10 Grid Search (24 experiments: 3 seeds × 2 buffer sizes × 4 param combinations)
./scripts/docker/run_seq_cifar10.sh

# CIFAR-100 Grid Search (24 experiments: 3 seeds × 2 buffer sizes × 4 param combinations)
./scripts/docker/run_seq_cifar100.sh

# TinyImageNet Grid Search (24 experiments: 3 seeds × 2 buffer sizes × 4 param combinations)
./scripts/docker/run_seq_tinyimg.sh
```

**AdaSARL Features:**
- GELU activation functions
- Balanced instance sampling
- Multiple buffer sizes (200, 500)
- Hyperparameter grid search for alpha, beta, op_weight, sm_weight
- Learning rate scheduling with warmup

### 🔍 AdaSARL Inference Grid Search Experiments

These experiments test AdaSARL with inference-only prototype guidance:

```bash
# CIFAR-10 Inference Grid Search (24 experiments: 3 seeds × 2 buffer sizes × 4 param combinations)
./scripts/docker/run_inference_seq_cifar10.sh

# CIFAR-100 Inference Grid Search (24 experiments: 3 seeds × 2 buffer sizes × 4 param combinations)
./scripts/docker/run_inference_seq_cifar100.sh

# TinyImageNet Inference Grid Search (24 experiments: 3 seeds × 2 buffer sizes × 4 param combinations)
./scripts/docker/run_inference_seq_tinyimg.sh
```

**AdaSARL Inference Features:**
- Prototype guidance only during evaluation (not training)
- Conservative guidance weights (0.05-0.1)
- High momentum prototype updates (0.99)
- All other AdaSARL features included

### 📈 Monitoring Experiments

#### Real-time Monitoring with TensorBoard

```bash
# Start TensorBoard for monitoring (run in separate terminal)
docker run --rm -p 6006:6006 \
    -v $(pwd)/outputs:/workspace/outputs \
    adasarl:latest \
    tensorboard --logdir=/workspace/outputs --host=0.0.0.0

# Then visit http://localhost:6006 in your browser
```

#### Check Progress

```bash
# Check running experiments
docker ps

# View logs of a running experiment
docker logs <container_name>

# Check output directories
ls -la outputs/experiments/adasarl/
ls -la outputs/experiments/adasarl_inference/
```

### Run AdaSARL Inference (Legacy Individual Runs)

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

#### GPU Not Detected

```bash
# Check if NVIDIA Docker is properly installed
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# If fails, install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Out of Memory Errors

```bash
# Monitor GPU memory usage:
watch -n 1 nvidia-smi

# Reduce batch size by editing the experiment scripts
# Or use smaller buffer sizes (200 instead of 500)
# Example: Edit scripts/adasarl/seq-cifar10.py to use smaller batch_size
```

#### Experiment Failures

```bash
# Check experiment logs
docker logs <container_name>

# Check if output directory has write permissions
sudo chmod -R 755 outputs/

# Clean up failed experiments and free space
docker system prune -f
docker volume prune -f
```

#### Slow Performance

```bash
# Use SSD storage for better I/O performance
# Ensure adequate RAM (16GB+ recommended)
# Monitor system resources:
htop

# Run fewer experiments in parallel
# Use smaller datasets for testing first
```

#### Docker Issues

```bash
# Permission issues
sudo chown -R $USER:$USER outputs/

# Port already in use
docker ps
docker stop <container_name>

# Docker not running
sudo systemctl start docker  # Linux
# Or start Docker Desktop on Windows/Mac
```

### 💡 Best Practices

#### Before Running Large Experiments

1. **Test with quick_test.sh first** - Verify setup works
2. **Ensure adequate disk space** - 50GB+ for full experiments
3. **Run single experiment first** - Test individual components
4. **Set up monitoring** - Use TensorBoard and system monitoring
5. **Consider running in batches** - Better resource management

#### Resource Management

```bash
# Run experiments sequentially for better resource control
./scripts/docker/run_seq_cifar10.sh
# Wait for completion, then:
./scripts/docker/run_seq_cifar100.sh

# Monitor disk usage regularly
df -h
du -sh outputs/

# Clean up intermediate files if needed
docker system prune -f
```

#### Data Backup and Analysis

```bash
# Backup important results
tar -czf adasarl_results_$(date +%Y%m%d).tar.gz outputs/experiments/

# Example result analysis
python -c "
import pandas as pd
import glob

# Compare buffer sizes for CIFAR-10
files = glob.glob('outputs/experiments/adasarl/adasarl-cifar10-**/results.csv', recursive=True)
results = []
for f in files:
    df = pd.read_csv(f)
    buffer_size = 200 if 'buf200' in f else 500
    df['buffer_size'] = buffer_size
    results.append(df)

if results:
    combined = pd.concat(results)
    print('Average accuracy by buffer size:')
    print(combined.groupby('buffer_size')['accuracy'].mean())
"
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

## � Available Docker Scripts

### 🚀 Master Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `run_all_experiments.sh` | Run all AdaSARL and Inference experiments | `./scripts/docker/run_all_experiments.sh` |
| `run_all_experiments.sh --adasarl-only` | Run only AdaSARL experiments | `./scripts/docker/run_all_experiments.sh --adasarl-only` |
| `run_all_experiments.sh --inference-only` | Run only Inference experiments | `./scripts/docker/run_all_experiments.sh --inference-only` |

### 🔬 AdaSARL Experiments (Full Training)

| Script | Dataset | Experiments | Description |
|--------|---------|-------------|-------------|
| `run_seq_cifar10.sh` | CIFAR-10 | 24 (3×2×4) | GELU + Balanced Sampling |
| `run_seq_cifar100.sh` | CIFAR-100 | 24 (3×2×4) | GELU + Balanced Sampling |
| `run_seq_tinyimg.sh` | TinyImageNet | 24 (3×2×4) | GELU + Balanced Sampling |

### 🔍 AdaSARL Inference Experiments (Inference-Only Prototypes)

| Script | Dataset | Experiments | Description |
|--------|---------|-------------|-------------|
| `run_inference_seq_cifar10.sh` | CIFAR-10 | 24 (3×2×4) | Conservative Prototype Guidance |
| `run_inference_seq_cifar100.sh` | CIFAR-100 | 24 (3×2×4) | Conservative Prototype Guidance |
| `run_inference_seq_tinyimg.sh` | TinyImageNet | 24 (3×2×4) | Conservative Prototype Guidance |

### 🧪 Utility Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `quick_test.sh` | Quick verification test | `./scripts/docker/quick_test.sh` |
| `start_tensorboard.sh` | Start TensorBoard monitoring | `./scripts/docker/start_tensorboard.sh` |

**Experiment Count Breakdown:**
- **3 seeds**: [1, 3, 5] for statistical significance
- **2 buffer sizes**: [200, 500] for different memory constraints  
- **4 hyperparameter combinations**: Different alpha/beta/op_weight/sm_weight/num_feats/balance_weight combinations
- **Total per dataset**: 24 experiments
- **Total for all experiments**: 144 experiments (6 scripts × 24 each)

## �📚 Additional Resources

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
