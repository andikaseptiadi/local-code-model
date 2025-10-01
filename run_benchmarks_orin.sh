#!/bin/bash
# Benchmark script for NVIDIA Jetson Orin NX
# Tests: CUDA, OpenBLAS, NEON, CPU naive/parallel

set -e

echo "================================================================"
echo "Jetson Orin NX Benchmark Suite"
echo "================================================================"
echo ""
echo "Platform: $(uname -m)"
echo "Kernel: $(uname -r)"
echo "CPU: $(nproc) cores"
echo ""

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA Version: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: CUDA not found"
fi

echo ""
echo "================================================================"

# Build with all backends
echo "Building with CGo enabled..."
export CGO_ENABLED=1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

go build -v

echo ""
echo "================================================================"
echo "Running comprehensive benchmarks..."
echo "================================================================"
echo ""

# Run all benchmarks with extended time
go test -bench . -benchtime=3s -timeout=30m 2>&1 | tee benchmark_orin_results.txt

echo ""
echo "================================================================"
echo "Key Comparisons (from benchmark_orin_results.txt)"
echo "================================================================"
echo ""

# Extract key results
echo "Matrix Multiplication (1024x1024):"
echo "-----------------------------------"
grep "BenchmarkG5GComparison" benchmark_orin_results.txt | grep 1024 || echo "No G5G comparison found"

echo ""
echo "CUDA Scaling:"
echo "-------------"
grep "BenchmarkCUDAScaling" benchmark_orin_results.txt || echo "No CUDA scaling found"

echo ""
echo "OpenBLAS vs Others:"
echo "-------------------"
grep "OpenBLAS" benchmark_orin_results.txt | head -5 || echo "No OpenBLAS results found"

echo ""
echo "NEON Performance:"
echo "-----------------"
grep "NEON" benchmark_orin_results.txt | head -5 || echo "No NEON results found"

echo ""
echo "================================================================"
echo "Benchmark complete! Results saved to: benchmark_orin_results.txt"
echo "================================================================"
