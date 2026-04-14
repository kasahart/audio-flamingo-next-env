# NVIDIA CUDA 12.8 base image with cuDNN for Blackwell GPU support (RTX PRO 6000)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3.11-distutils \
    git \
    git-lfs \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.7+ with CUDA 12.8 support (required for Blackwell architecture)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy and install Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Set environment variables for GPU inference
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/workspace
