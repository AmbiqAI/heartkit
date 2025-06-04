#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

sudo apt update
sudo apt install -y libopenblas-dev libyaml-dev ffmpeg wget ca-certificates

# # Install CUDA and cuDNN if not already installed
# if ! command -v nvcc &> /dev/null; then

#     CUDA_VERSION="12.3"
#     CUDNN_VERSION="8.9.7.29-1+cuda12.2" # Not sure why no 12.3

#     NVIDIA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
#     KEYRING_PACKAGE="cuda-keyring_1.1-1_all.deb"
#     KEYRING_PACKAGE_URL="$NVIDIA_REPO_URL/$KEYRING_PACKAGE"
#     KEYRING_PACKAGE_PATH="$(mktemp -d)"
#     KEYRING_PACKAGE_FILE="$KEYRING_PACKAGE_PATH/$KEYRING_PACKAGE"
#     wget -O "$KEYRING_PACKAGE_FILE" "$KEYRING_PACKAGE_URL"
#     sudo apt install -yq "$KEYRING_PACKAGE_FILE"
#     sudo apt update -yq

#     # Install CUDA libraries
#     cuda_pkg="cuda-libraries-${CUDA_VERSION/./-}"
#     sudo apt install -yq "$cuda_pkg"

#     # Install cuDNN
#     cudnn_pkg="libcudnn8=${CUDNN_VERSION}"
#     sudo apt install -yq "$cudnn_pkg_version"

#     # Install cuDNN dev
#     cudnn_dev_pkg="libcudnn8-dev=${CUDNN_VERSION}"
#     sudo apt install -yq "$cudnn_dev_pkg"

#     # Install  NVTX
#     nvtx_pkg="cuda-nvtx-${CUDA_VERSION/./-}"
#     sudo apt install -yq "$nvtx_pkg"

#     # Install CUDA Toolkit
#     toolkit_pkg="cuda-toolkit-${CUDA_VERSION/./-}"
#     sudo apt install -yq "$toolkit_pkg"

#     export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
#     export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#     # Clean up
#     sudo rm -rf /var/lib/apt/lists/*
# fi

# Install project dependencies
uv sync
