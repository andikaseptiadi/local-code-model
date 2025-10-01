# g5g Testing Guide

## Overview

This guide covers testing CUDA GPU acceleration on AWS g5g instances, which combine:
- **ARM Graviton2** (Neoverse N1) - 64-bit ARM CPUs
- **NVIDIA T4** Tensor Core GPU - Turing architecture

## Quick Start

### 1. Launch g5g Instance

```bash
# Recommended AMI: Amazon Linux 2023 ARM64
# Instance type: g5g.xlarge (minimum) or g5g.2xlarge (recommended)

aws ec2 run-instances \
  --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
  --instance-type g5g.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx
```

### 2. Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<instance-ip>

# Clone repository
git clone <your-repo>
cd local-code-model

# Run setup script
chmod +x setup_g5g.sh
./setup_g5g.sh

# If driver was installed, reboot
sudo reboot

# After reboot, verify NVIDIA driver
nvidia-smi
```

### 3. Build and Test

```bash
# Set environment
export CGO_ENABLED=1
export PATH=/usr/local/cuda/bin:/usr/local/go/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build
go build

# Run CUDA tests
go test -v -run TestCUDA

# Run all g5g backends comparison
go test -v -run TestCUDAAvailability
go test -v -bench BenchmarkG5GComparison
```

## Instance Types

| Type | T4 GPUs | vCPUs | RAM | GPU Memory | Use Case |
|------|---------|-------|-----|------------|----------|
| g5g.xlarge | 1 | 4 | 8 GB | 16 GB | Testing |
| g5g.2xlarge | 1 | 8 | 16 GB | 16 GB | Development |
| g5g.4xlarge | 1 | 16 | 32 GB | 16 GB | Benchmarking |
| g5g.8xlarge | 1 | 32 | 64 GB | 16 GB | Large workloads |
| g5g.16xlarge | 2 | 64 | 128 GB | 32 GB | Multi-GPU |

## NVIDIA T4 Specifications

- **Architecture**: Turing (Compute Capability 7.5)
- **CUDA Cores**: 2560 (40 SMs × 64 cores)
- **Tensor Cores**: 320
- **Memory**: 16 GB GDDR6
- **Memory Bandwidth**: 320 GB/s
- **Performance**:
  - FP64: 0.25 TFLOPS
  - FP32: 8.1 TFLOPS
  - FP16 (Tensor): 65 TFLOPS
  - INT8 (Tensor): 130 TOPS
- **Power**: 70W

## Expected Performance

### Matrix Multiplication (1024×1024, FP64)

| Backend | GFLOPS | vs Naive | Description |
|---------|--------|----------|-------------|
| Naive CPU | 0.15 | 1× | Pure Go, single-thread |
| NEON | 0.3 | 2× | ARM SIMD, 128-bit vectors |
| OpenBLAS | 3-5 | 20-30× | Optimized CPU BLAS |
| CUDA (T4) | 50-100 | 300-600× | GPU acceleration |

Note: CUDA performance scales dramatically with matrix size. Larger matrices (2048×2048) can achieve higher efficiency.

## Test Commands

### Availability Tests
```bash
# Check CUDA availability
go test -v -run TestCUDAAvailability

# Check device properties
go test -v -run TestCUDADeviceProperties

# Verify all backends
go test -v -run TestCUDAAvailability
go test -v -run TestOpenBLASAvailability
go test -v -run TestNEONCorrectness
```

### Correctness Tests
```bash
# CUDA correctness (various sizes)
go test -v -run TestCUDAMatMulCorrectness

# CUDA vs OpenBLAS comparison
go test -v -run TestCUDAvsOpenBLAS

# Memory limits test
go test -v -run TestCUDAMemoryLimits
```

### Performance Benchmarks
```bash
# CUDA scaling (64 to 4096)
go test -v -bench BenchmarkCUDAScaling

# All backends comparison
go test -v -bench BenchmarkG5GComparison

# CUDA only (detailed)
go test -v -bench BenchmarkCUDA
```

### Full Test Suite
```bash
# Run all tests and benchmarks
CGO_ENABLED=1 go test -v -bench . -benchtime=3s

# Generate performance profile
go test -bench BenchmarkG5GComparison -cpuprofile=cpu.prof
go tool pprof cpu.prof
```

## Troubleshooting

### NVIDIA Driver Not Found
```bash
# Check if driver is loaded
lsmod | grep nvidia

# Check if GPU is detected
lspci | grep -i nvidia

# Reinstall driver (Amazon Linux)
sudo yum install -y kernel-modules-nvidia
sudo reboot
```

### CUDA Not Found
```bash
# Check CUDA installation
ls -la /usr/local/cuda/

# Add to PATH if missing
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### Build Errors
```bash
# Ensure CGo is enabled
export CGO_ENABLED=1

# Check CUDA libraries
ls -la /usr/local/cuda/lib64/libcudart.so
ls -la /usr/local/cuda/lib64/libcublas.so

# Verify pkg-config can find libraries
pkg-config --list-all | grep cuda
```

### Runtime Errors
```bash
# Check GPU memory
nvidia-smi

# Monitor GPU utilization during test
watch -n 1 nvidia-smi

# Check CUDA errors
cuda-memcheck ./your-program
```

## Cost Optimization

### On-Demand Pricing (us-east-1)
- g5g.xlarge: ~$0.42/hour
- g5g.2xlarge: ~$0.67/hour
- g5g.4xlarge: ~$1.09/hour

### Spot Pricing
- Usually 50-70% cheaper than on-demand
- Good for testing/benchmarking

```bash
# Request spot instance
aws ec2 request-spot-instances \
  --spot-price "0.25" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

### Best Practices
1. Stop instances when not in use (storage ~$0.10/GB/month)
2. Use CloudWatch alarms to auto-stop idle instances
3. Consider Savings Plans if running long-term tests
4. Test on smaller instances first (g5g.xlarge) before scaling up

## Architecture Comparison

### g5g (ARM + GPU)
- **CPU**: Graviton2 (Neoverse N1), NEON only
- **GPU**: NVIDIA T4 (Turing)
- **Best for**: GPU-accelerated workloads, ARM testing
- **Limitation**: No SVE/SVE2 (Graviton2 doesn't have SVE)

### c7g (ARM CPU-only)
- **CPU**: Graviton3 (Neoverse V1), SVE 256-bit
- **Best for**: CPU-intensive, SVE testing
- **Limitation**: No GPU

### c8g (ARM CPU-only)
- **CPU**: Graviton4 (Neoverse V2), SVE2 128-bit
- **Best for**: Latest ARM features, SVE2 testing
- **Limitation**: No GPU

## References

- [AWS g5g Instances](https://aws.amazon.com/ec2/instance-types/g5g/)
- [NVIDIA T4 Datasheet](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
