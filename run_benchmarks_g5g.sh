#!/bin/bash
# Benchmark script for AWS g5g (Graviton2 + NVIDIA T4G)
# Tests: CUDA (T4), OpenBLAS, NEON, CPU naive/parallel

set -e

echo "================================================================"
echo "AWS g5g Instance Benchmark Suite"
echo "================================================================"
echo ""

# CPU info
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
echo "CPU: $CPU_MODEL (Graviton2)"
echo "Cores: $(nproc)"
echo ""

# GPU info
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    echo ""
    echo "CUDA Version: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
else
    echo "ERROR: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

echo ""
echo "Expected Performance (1024x1024 FP64):"
echo "  CPU (OpenBLAS): 30-40 GFLOPS"
echo "  GPU (T4): 150-200 GFLOPS"
echo ""
echo "================================================================"

# Build
echo "Building with CUDA support..."
export CGO_ENABLED=1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

go build -v

echo ""
echo "================================================================"
echo "Running comprehensive benchmarks..."
echo "================================================================"
echo ""

# Run all benchmarks
go test -bench . -benchtime=3s -timeout=30m 2>&1 | tee benchmark_g5g_results.txt

echo ""
echo "================================================================"
echo "Key Results - CPU vs GPU Comparison"
echo "================================================================"
echo ""

echo "G5G Comparison Benchmark (1024x1024):"
echo "--------------------------------------"
grep "BenchmarkG5GComparison" benchmark_g5g_results.txt || echo "No G5G comparison found"

echo ""
echo "CUDA Scaling Across Matrix Sizes:"
echo "----------------------------------"
grep "BenchmarkCUDAScaling" benchmark_g5g_results.txt || echo "No CUDA scaling found"

echo ""
echo "OpenBLAS Performance:"
echo "--------------------"
grep "OpenBLAS" benchmark_g5g_results.txt | head -5 || echo "No OpenBLAS results"

echo ""
echo "NEON Performance:"
echo "-----------------"
grep "NEON" benchmark_g5g_results.txt | head -5 || echo "No NEON results"

echo ""
echo "================================================================"
echo "Performance Analysis"
echo "================================================================"
echo ""

# Calculate GFLOPS for common sizes
echo "Calculating GFLOPS for key operations..."
echo ""

# 1024x1024 matmul: 2 * 1024^3 = 2,147,483,648 FLOPs
echo "1024x1024 Matrix Multiplication (2.15 GFLOPs per operation):"
echo "-------------------------------------------------------------"
grep "1024" benchmark_g5g_results.txt | while read -r line; do
    # Extract ns/op
    ns_per_op=$(echo "$line" | awk '{print $(NF-1)}')
    if [[ "$ns_per_op" =~ ^[0-9]+$ ]]; then
        # Calculate GFLOPS: 2.147483648 / (ns_per_op / 1e9)
        gflops=$(echo "scale=2; 2147483648 / $ns_per_op" | bc)
        echo "  $line -> ${gflops} GFLOPS"
    fi
done

echo ""
echo "================================================================"
echo "Benchmark complete! Results saved to: benchmark_g5g_results.txt"
echo "================================================================"
echo ""
echo "T4 should significantly outperform Graviton2 + OpenBLAS on FP64."
echo "If not, check:"
echo "  - GPU utilization (nvidia-smi)"
echo "  - CUDA kernel configuration"
echo "  - Data transfer overhead"
