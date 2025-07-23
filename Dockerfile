# AdaSARL: Adaptive Domain-Agnostic Semantic Representation for Continual Learning
# Dockerfile for running AdaSARL experiments

FROM pytorch/pytorch:2.7.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /workspace/adasarl

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for better performance
RUN pip install --no-cache-dir \
    tensorboard \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn \
    opencv-python \
    pillow \
    jupyter \
    ipywidgets

# Copy the entire AdaSARL codebase
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/adasarl/outputs \
    && mkdir -p /workspace/adasarl/logs \
    && mkdir -p /workspace/adasarl/checkpoints \
    && mkdir -p /workspace/adasarl/data

# Set permissions
RUN chmod +x /workspace/adasarl/main.py

# Expose port for TensorBoard
EXPOSE 6006

# Default command
CMD ["bash"] 