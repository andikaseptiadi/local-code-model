// +build darwin,cgo

package main

import (
	"fmt"
	"math"
	"testing"
)

// TestMetalAvailability tests if Metal can be initialized.
func TestMetalAvailability(t *testing.T) {
	backend, err := NewMetalBackend()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}

	if !backend.IsAvailable() {
		t.Error("Metal backend created but reports not available")
	}

	t.Logf("Metal device: %s", backend.DeviceName())
}

// TestMetalMatMul tests basic matrix multiplication on Metal.
func TestMetalMatMul(t *testing.T) {
	backend, err := NewMetalBackend()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}

	// Test small matrix
	a := NewTensor(4, 4)
	b := NewTensor(4, 4)

	// Fill with simple values for easy verification
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			a.Set(float64(i+1), i, j)
			b.Set(float64(j+1), i, j)
		}
	}

	result, err := backend.MatMul(a, b)
	if err != nil {
		t.Fatalf("Metal MatMul failed: %v", err)
	}

	// Verify dimensions
	if len(result.shape) != 2 || result.shape[0] != 4 || result.shape[1] != 4 {
		t.Errorf("Result shape mismatch: got %v, want [4 4]", result.shape)
	}

	// Verify it's not all zeros (sanity check)
	allZero := true
	for i := range result.data {
		if result.data[i] != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("Result is all zeros (GPU computation likely failed)")
	}

	t.Logf("Metal MatMul result sample: %.2f", result.At(0, 0))
}

// TestMetalVsCPU compares Metal and CPU results for correctness.
func TestMetalVsCPU(t *testing.T) {
	backend, err := NewMetalBackend()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}

	sizes := []int{32, 64, 128}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// CPU result (ground truth)
			cpuConfig := CPUOnlyConfig()
			cpuResult := MatMulWithStrategy(a, b, StrategyNaive, cpuConfig)

			// Metal result
			metalResult, err := backend.MatMul(a, b)
			if err != nil {
				t.Fatalf("Metal MatMul failed: %v", err)
			}

			// Compare with tolerance (fp32 on GPU vs fp64 on CPU)
			// Use larger tolerance due to precision difference
			if !tensorsEqualApprox(cpuResult, metalResult, 1e-4) {
				t.Errorf("Metal result differs from CPU")

				// Show first mismatch for debugging
				for i := 0; i < min(size, 4); i++ {
					for j := 0; j < min(size, 4); j++ {
						cpu := cpuResult.At(i, j)
						gpu := metalResult.At(i, j)
						diff := math.Abs(cpu - gpu)
						if diff > 1e-4 {
							t.Logf("First mismatch at [%d,%d]: CPU=%.6f, GPU=%.6f, diff=%.6f",
								i, j, cpu, gpu, diff)
							return
						}
					}
				}
			}
		})
	}
}

// BenchmarkMetalVsCPU compares Metal and CPU performance.
func BenchmarkMetalVsCPU(b *testing.B) {
	backend, err := NewMetalBackend()
	if err != nil {
		b.Skipf("Metal not available: %v", err)
	}

	sizes := []int{128, 256, 512, 1024}

	for _, size := range sizes {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		// Benchmark CPU (cache-blocked parallel)
		b.Run(fmt.Sprintf("CPU_%d", size), func(b *testing.B) {
			cfg := DefaultBackendConfig()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithStrategy(a, mat, StrategyCacheBlocked, cfg)
			}
		})

		// Benchmark Metal GPU
		b.Run(fmt.Sprintf("Metal_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := backend.MatMul(a, mat)
				if err != nil {
					b.Fatalf("Metal MatMul failed: %v", err)
				}
			}
		})
	}
}

// TestMetalErrorHandling tests error conditions.
func TestMetalErrorHandling(t *testing.T) {
	backend, err := NewMetalBackend()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}

	// Test dimension mismatch
	a := NewTensor(4, 5)
	b := NewTensor(3, 6) // Incompatible: a.cols != b.rows

	_, err = backend.MatMul(a, b)
	if err == nil {
		t.Error("Expected error for incompatible dimensions, got nil")
	}

	// Test non-2D tensors
	a3d := NewTensor(2, 3, 4)
	b2d := NewTensor(3, 4)

	_, err = backend.MatMul(a3d, b2d)
	if err == nil {
		t.Error("Expected error for 3D tensor, got nil")
	}
}

// TestMetalLargeMatrix tests Metal with large matrices.
func TestMetalLargeMatrix(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large matrix test in short mode")
	}

	backend, err := NewMetalBackend()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}

	// Test large matrix (should show GPU advantage)
	size := 2048
	a := NewTensorRand(size, size)
	b := NewTensorRand(size, size)

	result, err := backend.MatMul(a, b)
	if err != nil {
		t.Fatalf("Metal MatMul failed for large matrix: %v", err)
	}

	// Verify shape
	if result.shape[0] != size || result.shape[1] != size {
		t.Errorf("Result shape mismatch: got %v, want [%d %d]", result.shape, size, size)
	}

	t.Logf("Successfully computed %dx%d matrix multiplication on Metal", size, size)
}
