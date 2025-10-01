#!/bin/bash
# Benchmark script for AWS Graviton (2/3/3E/4)
# Tests: SVE/SVE2, OpenBLAS, NEON, CPU naive/parallel

set -e

echo "================================================================"
echo "AWS Graviton Benchmark Suite"
echo "================================================================"
echo ""

# Detect Graviton generation
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
echo "CPU: $CPU_MODEL"
echo "Cores: $(nproc)"
echo "Architecture: $(uname -m)"
echo ""

# Check for SVE/SVE2
echo "Vector Extensions:"
if grep -q "sve" /proc/cpuinfo; then
    echo "  ✓ SVE detected"
    SVE_VL=$(cat /proc/sys/abi/sve_default_vector_length 2>/dev/null || echo "unknown")
    echo "    Vector length: $SVE_VL bits"
fi

if grep -q "sve2" /proc/cpuinfo; then
    echo "  ✓ SVE2 detected"
fi

if ! grep -q "sve" /proc/cpuinfo; then
    echo "  ✓ NEON only (Graviton2)"
fi

echo ""

# Detect generation based on CPU model
if [[ "$CPU_MODEL" == *"Neoverse-N1"* ]]; then
    GENERATION="Graviton2"
    EXPECTED_VL="NEON (128-bit)"
    EXPECTED_PERF="30-40 GFLOPS (OpenBLAS)"
elif [[ "$CPU_MODEL" == *"Neoverse-V1"* ]]; then
    GENERATION="Graviton3/3E"
    EXPECTED_VL="SVE (256-bit, 2× engines)"
    EXPECTED_PERF="60-80 GFLOPS (OpenBLAS + SVE)"
elif [[ "$CPU_MODEL" == *"Neoverse-V2"* ]]; then
    GENERATION="Graviton4"
    EXPECTED_VL="SVE2 (128-bit, 4× engines)"
    EXPECTED_PERF="80-100 GFLOPS (OpenBLAS + SVE2)"
else
    GENERATION="Unknown"
    EXPECTED_VL="Unknown"
    EXPECTED_PERF="Unknown"
fi

echo "Detected Generation: $GENERATION"
echo "Expected Vector Length: $EXPECTED_VL"
echo "Expected Performance: $EXPECTED_PERF"
echo ""
echo "================================================================"

# Build
echo "Building with CGo enabled..."
export CGO_ENABLED=1

go build -v

echo ""
echo "================================================================"
echo "Running comprehensive benchmarks..."
echo "================================================================"
echo ""

# Run all benchmarks
go test -bench . -benchtime=3s -timeout=30m 2>&1 | tee "benchmark_graviton_${GENERATION}_results.txt"

echo ""
echo "================================================================"
echo "Key Results for $GENERATION"
echo "================================================================"
echo ""

RESULT_FILE="benchmark_graviton_${GENERATION}_results.txt"

echo "OpenBLAS Performance:"
echo "--------------------"
grep "OpenBLAS" "$RESULT_FILE" | head -5 || echo "No OpenBLAS results"

echo ""
echo "SVE/NEON Performance:"
echo "--------------------"
if [[ "$GENERATION" == "Graviton2" ]]; then
    grep "NEON" "$RESULT_FILE" | head -5 || echo "No NEON results"
else
    grep "SVE" "$RESULT_FILE" | head -5 || echo "No SVE results"
fi

echo ""
echo "Graviton-Specific Tests:"
echo "------------------------"
grep "Graviton" "$RESULT_FILE" || echo "No Graviton-specific tests"

echo ""
echo "================================================================"
echo "Benchmark complete! Results saved to: $RESULT_FILE"
echo "================================================================"
