#!/bin/bash
# ===========================================================================
# Setup script for AWS g5g instances (Graviton2 + NVIDIA T4G)
# ===========================================================================
#
# This script installs all dependencies needed to run the tests on g5g:
#   - NVIDIA drivers
#   - CUDA toolkit
#   - cuBLAS library
#   - OpenBLAS
#   - Go compiler
#
# Usage:
#   ./setup_g5g.sh
#
# AMI Recommendation:
#   - Amazon Linux 2023 (AL2023) ARM64 - NVIDIA drivers pre-installed
#   - Deep Learning AMI GPU - CUDA pre-installed (easiest option)
#
# Instance Types:
#   - g5g.xlarge:  1× T4, 4 vCPUs, 8GB RAM  (testing)
#   - g5g.2xlarge: 1× T4, 8 vCPUs, 16GB RAM (development)
#   - g5g.4xlarge: 1× T4, 16 vCPUs, 32GB RAM (benchmarking)
#
# ===========================================================================

set -e  # Exit on error

echo "=========================================="
echo "g5g Setup for CUDA + Graviton2 Testing"
echo "=========================================="
echo ""

# Check if we're on ARM64
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "❌ Error: This script is for ARM64 (aarch64) only"
    echo "   Detected architecture: $ARCH"
    exit 1
fi

echo "✓ Architecture: $ARCH (ARM64)"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
    echo "✓ OS: $NAME $VERSION"
else
    echo "❌ Cannot detect OS"
    exit 1
fi
echo ""

# Check if NVIDIA GPU is present
if ! lspci | grep -i nvidia > /dev/null 2>&1; then
    echo "⚠️  Warning: No NVIDIA GPU detected"
    echo "   This script is designed for g5g instances with T4 GPUs"
    echo "   Continuing anyway..."
    echo ""
fi

# Update package lists based on OS
echo "📦 Updating package lists..."
if [ "$OS" = "amzn" ]; then
    # Amazon Linux
    sudo yum update -y -q
    INSTALL_CMD="sudo yum install -y -q"
elif [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    # Ubuntu/Debian
    sudo apt-get update -qq
    INSTALL_CMD="sudo apt-get install -y -qq"
else
    echo "❌ Unsupported OS: $OS"
    exit 1
fi

# Install basic build tools
echo "📦 Installing build tools..."
if [ "$OS" = "amzn" ]; then
    $INSTALL_CMD \
        gcc \
        gcc-c++ \
        make \
        pkgconfig \
        wget \
        curl \
        git \
        tar
else
    $INSTALL_CMD \
        build-essential \
        pkg-config \
        wget \
        curl \
        git
fi

# Install NVIDIA driver (if not already installed)
echo "📦 Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "   Installing NVIDIA driver..."
    if [ "$OS" = "amzn" ]; then
        # Amazon Linux - driver usually pre-installed, but install just in case
        $INSTALL_CMD kernel-modules-nvidia
        sudo nvidia-modprobe
    else
        # Ubuntu
        $INSTALL_CMD nvidia-driver-535
    fi
    echo "   ⚠️  NVIDIA driver installed - REBOOT REQUIRED"
    echo "   Run 'sudo reboot' and re-run this script after reboot"
else
    echo "   ✓ NVIDIA driver already installed"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "   (nvidia-smi exists but may need reboot)"
fi

# Install CUDA toolkit
echo "📦 Checking CUDA toolkit..."
if ! command -v nvcc &> /dev/null; then
    echo "   Installing CUDA toolkit..."

    if [ "$OS" = "amzn" ]; then
        # Amazon Linux 2023
        $INSTALL_CMD cuda-toolkit
        # CUDA is installed to /usr/local/cuda on AL2023
    else
        # Ubuntu 22.04 ARM64
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        rm cuda-keyring_1.1-1_all.deb

        sudo apt-get update -qq
        $INSTALL_CMD cuda-toolkit-12-3
    fi

    # Add CUDA to PATH
    if ! grep -q "cuda/bin" ~/.bashrc 2>/dev/null; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    echo "   ✓ CUDA toolkit installed"
else
    echo "   ✓ CUDA toolkit already installed"
    nvcc --version 2>/dev/null | grep "release" || echo "   (nvcc found but may need path setup)"
fi

# Install OpenBLAS
echo "📦 Installing OpenBLAS..."
if [ "$OS" = "amzn" ]; then
    $INSTALL_CMD openblas-devel
else
    $INSTALL_CMD libopenblas-dev
fi
echo "   ✓ OpenBLAS installed"

# Install Go
echo "📦 Checking Go..."
if ! command -v go &> /dev/null; then
    echo "   Installing Go 1.23..."
    GO_VERSION="1.23.0"
    wget -q https://go.dev/dl/go${GO_VERSION}.linux-arm64.tar.gz
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-arm64.tar.gz
    rm go${GO_VERSION}.linux-arm64.tar.gz

    if ! grep -q "/usr/local/go/bin" ~/.bashrc 2>/dev/null; then
        echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
    fi
    export PATH=/usr/local/go/bin:$PATH

    echo "   ✓ Go installed"
else
    echo "   ✓ Go already installed"
    go version
fi

# Verify installations
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

echo "🔍 Checking installations..."
echo ""

# Check NVIDIA driver
echo "NVIDIA Driver:"
if nvidia-smi &> /dev/null; then
    echo "  ✅ nvidia-smi works"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "  ❌ nvidia-smi not working (reboot may be needed)"
fi
echo ""

# Check CUDA
echo "CUDA:"
if [ -f /usr/local/cuda/lib64/libcudart.so ]; then
    echo "  ✅ CUDA runtime library found"
    ls -lh /usr/local/cuda/lib64/libcudart.so
else
    echo "  ❌ CUDA runtime library not found"
fi
echo ""

# Check cuBLAS
echo "cuBLAS:"
if [ -f /usr/local/cuda/lib64/libcublas.so ]; then
    echo "  ✅ cuBLAS library found"
    ls -lh /usr/local/cuda/lib64/libcublas.so
else
    echo "  ❌ cuBLAS library not found"
fi
echo ""

# Check OpenBLAS
echo "OpenBLAS:"
if [ -f /usr/lib/aarch64-linux-gnu/libopenblas.so ]; then
    echo "  ✅ OpenBLAS library found"
    ls -lh /usr/lib/aarch64-linux-gnu/libopenblas.so
else
    echo "  ❌ OpenBLAS library not found"
fi
echo ""

# Check Go
echo "Go:"
if command -v go &> /dev/null; then
    echo "  ✅ Go compiler found"
    go version
else
    echo "  ❌ Go compiler not found"
fi
echo ""

# Build test
echo "=========================================="
echo "Build Test"
echo "=========================================="
echo ""

if [ -f "tensor.go" ]; then
    echo "📦 Building project..."
    export CGO_ENABLED=1
    export PATH=/usr/local/cuda/bin:/usr/local/go/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    if go build -v .; then
        echo "  ✅ Build successful"
    else
        echo "  ❌ Build failed"
        exit 1
    fi
    echo ""

    echo "🧪 Running quick test..."
    if go test -v -run TestCUDAAvailability .; then
        echo "  ✅ CUDA test passed"
    else
        echo "  ⚠️  CUDA test failed (may need reboot if driver was just installed)"
    fi
else
    echo "⚠️  Not in project directory (tensor.go not found)"
    echo "   Navigate to project directory and run: go build"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. If driver was just installed: sudo reboot"
echo "  2. Build project: CGO_ENABLED=1 go build"
echo "  3. Run tests: CGO_ENABLED=1 go test -v -run CUDA"
echo "  4. Run benchmarks: CGO_ENABLED=1 go test -v -bench BenchmarkG5GComparison"
echo ""
echo "Environment variables (add to ~/.bashrc):"
echo "  export PATH=/usr/local/cuda/bin:/usr/local/go/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo "  export CGO_ENABLED=1"
echo ""
