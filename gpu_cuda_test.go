//go:build linux && cgo

package main

import (
	"fmt"
	"math"
	"testing"
)

// ===========================================================================
// CUDA GPU Tests for AWS g5g (NVIDIA T4G)
// ===========================================================================
//
// g5g instances feature NVIDIA T4 Tensor Core GPUs with ARM Graviton2:
//   - g5g.xlarge:  1× T4 (16GB), 4 vCPUs, 8GB RAM
//   - g5g.2xlarge: 1× T4 (16GB), 8 vCPUs, 16GB RAM
//   - g5g.4xlarge: 1× T4 (16GB), 16 vCPUs, 32GB RAM
//   - g5g.8xlarge: 1× T4 (16GB), 32 vCPUs, 64GB RAM
//   - g5g.16xlarge: 2× T4 (32GB), 64 vCPUs, 128GB RAM
//
// NVIDIA T4 Specifications:
//   - Turing Architecture (compute capability 7.5)
//   - 2560 CUDA cores
//   - 320 Tensor Cores
//   - 16GB GDDR6 memory (320 GB/s bandwidth)
//   - FP32: 8.1 TFLOPS
//   - FP16 (Tensor Cores): 65 TFLOPS
//   - INT8 (Tensor Cores): 130 TOPS
//   - Power: 70W

// TestCUDAAvailability tests CUDA/GPU availability
func TestCUDAAvailability(t *testing.T) {
	backend, err := NewCUDABackend()

	if err != nil {
		t.Logf("⚠️  CUDA not available: %v", err)
		t.Logf("To use CUDA on g5g:")
		t.Logf("  1. Launch g5g instance (g5g.xlarge or larger)")
		t.Logf("  2. Install NVIDIA drivers: sudo apt install nvidia-driver-XXX")
		t.Logf("  3. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit")
		t.Logf("  4. Verify: nvidia-smi")
		t.Skip("CUDA not available")
	}

	if backend == nil {
		t.Fatal("CUDA backend should exist")
	}

	if !backend.IsAvailable() {
		t.Skip("CUDA reports as unavailable")
	}

	t.Logf("✅ CUDA available: %s", backend.DeviceName())
	t.Logf("Info:\n%s", backend.GetInfo())
}

// TestCUDADeviceProperties tests CUDA device detection
func TestCUDADeviceProperties(t *testing.T) {
	backend, err := NewCUDABackend()
	if err != nil || !backend.IsAvailable() {
		t.Skip("CUDA not available")
	}

	t.Logf("=== CUDA Device Properties ===")
	t.Logf("Device: %s", backend.props.Name)
	t.Logf("Memory: %.2f GB", float64(backend.props.TotalGlobalMem)/(1024*1024*1024))
	t.Logf("Compute Capability: %s", backend.props.ComputeCapability)
	t.Logf("SM Count: %d", backend.props.MultiProcessorCount)
	t.Logf("Clock Rate: %.2f GHz", float64(backend.props.ClockRate)/1e6)
	t.Logf("Shared Memory/Block: %d KB", backend.props.SharedMemPerBlock/1024)

	// Verify expected properties for T4
	if backend.props.ComputeCapability == "7.5" {
		t.Logf("✓ Detected NVIDIA T4 (Turing, compute capability 7.5)")

		// T4 has 40 SMs
		if backend.props.MultiProcessorCount != 40 {
			t.Logf("⚠️  Expected 40 SMs for T4, got %d", backend.props.MultiProcessorCount)
		}
	}

	// Verify minimum memory (T4 has 16GB)
	minMemoryGB := 15.0 // Allow some overhead
	actualMemoryGB := float64(backend.props.TotalGlobalMem) / (1024 * 1024 * 1024)
	if actualMemoryGB < minMemoryGB {
		t.Errorf("Expected at least %.1f GB memory, got %.2f GB", minMemoryGB, actualMemoryGB)
	}
}

// TestCUDAMatMulCorrectness tests CUDA matrix multiplication correctness
func TestCUDAMatMulCorrectness(t *testing.T) {
	backend, err := NewCUDABackend()
	if err != nil || !backend.IsAvailable() {
		t.Skip("CUDA not available")
	}
	defer backend.Close()

	sizes := []int{4, 8, 16, 32, 64, 128, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// CUDA result
			result, err := backend.MatMul(a, b)
			if err != nil {
				t.Fatalf("CUDA MatMul failed: %v", err)
			}

			// Reference result (CPU)
			expected := MatMul(a, b)

			// Compare
			maxDiff := 0.0
			avgDiff := 0.0
			for i := range result.data {
				diff := math.Abs(result.data[i] - expected.data[i])
				if diff > maxDiff {
					maxDiff = diff
				}
				avgDiff += diff
			}
			avgDiff /= float64(len(result.data))

			// CUDA should be very accurate (within floating point precision)
			tolerance := 1e-10
			if maxDiff > tolerance {
				t.Errorf("CUDA result differs: max diff %e, avg diff %e", maxDiff, avgDiff)
			}

			if size <= 64 {
				t.Logf("Size %d: max diff %e, avg diff %e ✅", size, maxDiff, avgDiff)
			}
		})
	}
}

// TestCUDAvsOpenBLAS compares CUDA with OpenBLAS for correctness
func TestCUDAvsOpenBLAS(t *testing.T) {
	cuda, err := NewCUDABackend()
	if err != nil || !cuda.IsAvailable() {
		t.Skip("CUDA not available")
	}
	defer cuda.Close()

	openblas, err := NewOpenBLASBackend()
	if err != nil || !openblas.IsAvailable() {
		t.Skip("OpenBLAS not available")
	}

	size := 256
	a := NewTensorRand(size, size)
	b := NewTensorRand(size, size)

	// CUDA result
	cudaResult, err := cuda.MatMul(a, b)
	if err != nil {
		t.Fatalf("CUDA MatMul failed: %v", err)
	}

	// OpenBLAS result
	blasResult, err := openblas.MatMul(a, b)
	if err != nil {
		t.Fatalf("OpenBLAS MatMul failed: %v", err)
	}

	// Compare
	maxDiff := 0.0
	for i := range cudaResult.data {
		diff := math.Abs(cudaResult.data[i] - blasResult.data[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tolerance := 1e-10
	if maxDiff > tolerance {
		t.Errorf("CUDA and OpenBLAS differ: max diff %e", maxDiff)
	}

	t.Logf("CUDA vs OpenBLAS (size %d): max diff %e ✅", size, maxDiff)
}

// BenchmarkCUDA benchmarks CUDA performance
func BenchmarkCUDA(b *testing.B) {
	backend, err := NewCUDABackend()
	if err != nil || !backend.IsAvailable() {
		b.Skip("CUDA not available")
	}
	defer backend.Close()

	sizes := []int{128, 256, 512, 1024, 2048}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}

			// Calculate GFLOPS
			ops := 2.0 * float64(size) * float64(size) * float64(size)
			seconds := b.Elapsed().Seconds()
			gflops := (ops * float64(b.N)) / seconds / 1e9

			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkG5GComparison compares all backends available on g5g
func BenchmarkG5GComparison(b *testing.B) {
	size := 1024

	b.Logf("=== g5g Backend Comparison (size %d×%d) ===", size, size)
	b.Logf("Expected backends:")
	b.Logf("  - CPU: Graviton2 (ARM Neoverse N1)")
	b.Logf("  - NEON: 128-bit SIMD")
	b.Logf("  - OpenBLAS: Optimized BLAS")
	b.Logf("  - CUDA: NVIDIA T4 GPU")

	// Naive CPU
	b.Run("Naive", func(b *testing.B) {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = MatMul(a, mat)
		}

		ops := 2.0 * float64(size) * float64(size) * float64(size)
		gflops := (ops * float64(b.N)) / b.Elapsed().Seconds() / 1e9
		b.ReportMetric(gflops, "GFLOPS")
	})

	// NEON
	if IsSIMDAvailable() {
		b.Run("NEON", func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulSIMD(a, mat)
			}

			ops := 2.0 * float64(size) * float64(size) * float64(size)
			gflops := (ops * float64(b.N)) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}

	// OpenBLAS
	openblas, err := NewOpenBLASBackend()
	if err == nil && openblas.IsAvailable() {
		b.Run("OpenBLAS", func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = openblas.MatMul(a, mat)
			}

			ops := 2.0 * float64(size) * float64(size) * float64(size)
			gflops := (ops * float64(b.N)) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}

	// CUDA
	cuda, err := NewCUDABackend()
	if err == nil && cuda.IsAvailable() {
		defer cuda.Close()
		b.Run("CUDA", func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = cuda.MatMul(a, mat)
			}

			ops := 2.0 * float64(size) * float64(size) * float64(size)
			gflops := (ops * float64(b.N)) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkCUDAScaling tests CUDA performance scaling
func BenchmarkCUDAScaling(b *testing.B) {
	backend, err := NewCUDABackend()
	if err != nil || !backend.IsAvailable() {
		b.Skip("CUDA not available")
	}
	defer backend.Close()

	b.Logf("=== CUDA Performance Scaling ===")
	b.Logf("Device: %s", backend.DeviceName())
	b.Logf("T4 Specifications:")
	b.Logf("  - 2560 CUDA cores (40 SMs × 64 cores)")
	b.Logf("  - 320 Tensor Cores")
	b.Logf("  - FP64: 0.25 TFLOPS")
	b.Logf("  - FP32: 8.1 TFLOPS")
	b.Logf("  - FP16 (Tensor): 65 TFLOPS")
	b.Logf("  - Memory Bandwidth: 320 GB/s")

	sizes := []int{64, 128, 256, 512, 1024, 2048, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}

			ops := 2.0 * float64(size) * float64(size) * float64(size)
			seconds := b.Elapsed().Seconds()
			gflops := (ops * float64(b.N)) / seconds / 1e9

			// Calculate efficiency vs T4 FP64 peak (0.25 TFLOPS)
			peakTFLOPS := 0.25
			efficiency := (gflops / 1000.0) / peakTFLOPS * 100.0

			b.ReportMetric(gflops, "GFLOPS")
			b.ReportMetric(efficiency, "%peak")
		})
	}
}

// TestCUDAMemoryLimits tests CUDA with large matrices
func TestCUDAMemoryLimits(t *testing.T) {
	backend, err := NewCUDABackend()
	if err != nil || !backend.IsAvailable() {
		t.Skip("CUDA not available")
	}
	defer backend.Close()

	// T4 has 16GB memory
	// Test progressively larger matrices until we approach limits
	sizes := []int{1024, 2048, 4096, 8192}

	for _, size := range sizes {
		memoryNeeded := 3 * size * size * 8 / (1024 * 1024 * 1024) // 3 matrices × 8 bytes × size^2 → GB

		t.Run(fmt.Sprintf("size_%d_%.2fGB", size, float64(memoryNeeded)), func(t *testing.T) {
			t.Logf("Matrix size: %d×%d, memory needed: %.2f GB", size, size, float64(memoryNeeded))

			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			result, err := backend.MatMul(a, b)
			if err != nil {
				t.Logf("⚠️  Failed at size %d: %v", size, err)
				return
			}

			// Quick sanity check
			if result.shape[0] != size || result.shape[1] != size {
				t.Errorf("Result shape mismatch: got %v, expected [%d %d]",
					result.shape, size, size)
			}

			t.Logf("✅ Success at size %d×%d (%.2f GB)", size, size, float64(memoryNeeded))
		})
	}
}
