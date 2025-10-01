// +build darwin,cgo

package main

import (
	"fmt"
	"testing"
)

// TestAccelerateAvailability tests if Accelerate is available.
func TestAccelerateAvailability(t *testing.T) {
	backend, err := NewAccelerateBackend()
	if err != nil {
		t.Fatalf("Accelerate should always be available on macOS: %v", err)
	}

	if !backend.IsAvailable() {
		t.Error("Accelerate backend created but reports not available")
	}

	t.Logf("Accelerate backend: %s", backend.Name())
}

// TestAccelerateMatMul tests basic matrix multiplication with Accelerate.
func TestAccelerateMatMul(t *testing.T) {
	backend, err := NewAccelerateBackend()
	if err != nil {
		t.Fatalf("Failed to create Accelerate backend: %v", err)
	}

	// Test small matrix
	a := NewTensor(4, 4)
	b := NewTensor(4, 4)

	// Fill with simple values
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			a.Set(float64(i+1), i, j)
			b.Set(float64(j+1), i, j)
		}
	}

	result, err := backend.MatMul(a, b)
	if err != nil {
		t.Fatalf("Accelerate MatMul failed: %v", err)
	}

	// Verify dimensions
	if len(result.shape) != 2 || result.shape[0] != 4 || result.shape[1] != 4 {
		t.Errorf("Result shape mismatch: got %v, want [4 4]", result.shape)
	}

	t.Logf("Accelerate MatMul result[0,0]: %.2f", result.At(0, 0))
}

// TestAccelerateVsCPU compares Accelerate and CPU results for correctness.
func TestAccelerateVsCPU(t *testing.T) {
	backend, err := NewAccelerateBackend()
	if err != nil {
		t.Fatalf("Failed to create Accelerate backend: %v", err)
	}

	sizes := []int{32, 64, 128, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// CPU result (ground truth)
			cpuConfig := CPUOnlyConfig()
			cpuResult := MatMulWithStrategy(a, b, StrategyNaive, cpuConfig)

			// Accelerate result
			accelResult, err := backend.MatMul(a, b)
			if err != nil {
				t.Fatalf("Accelerate MatMul failed: %v", err)
			}

			// Compare (should be very close, both use fp64)
			if !tensorsEqualApprox(cpuResult, accelResult, 1e-10) {
				t.Errorf("Accelerate result differs from CPU")
			}
		})
	}
}

// TestAccelerateFloat32 tests the fp32 version.
func TestAccelerateFloat32(t *testing.T) {
	backend, err := NewAccelerateBackend()
	if err != nil {
		t.Fatalf("Failed to create Accelerate backend: %v", err)
	}

	a := NewTensorRand(64, 64)
	b := NewTensorRand(64, 64)

	// CPU reference (fp64)
	cpuResult := MatMul(a, b)

	// Accelerate fp32
	accelResult, err := backend.MatMulFloat32(a, b)
	if err != nil {
		t.Fatalf("Accelerate MatMulFloat32 failed: %v", err)
	}

	// Compare with larger tolerance (fp32 vs fp64)
	if !tensorsEqualApprox(cpuResult, accelResult, 1e-4) {
		t.Errorf("Accelerate fp32 result differs significantly from fp64 CPU")
	}
}

// BenchmarkAccelerateVsCPU compares Accelerate and CPU performance.
func BenchmarkAccelerateVsCPU(b *testing.B) {
	backend, err := NewAccelerateBackend()
	if err != nil {
		b.Skipf("Accelerate not available: %v", err)
	}

	sizes := []int{128, 256, 512, 1024}

	for _, size := range sizes {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		// Benchmark CPU (naive)
		b.Run(fmt.Sprintf("CPU_Naive_%d", size), func(b *testing.B) {
			cfg := CPUOnlyConfig()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithStrategy(a, mat, StrategyNaive, cfg)
			}
		})

		// Benchmark CPU (cache-blocked)
		b.Run(fmt.Sprintf("CPU_Cached_%d", size), func(b *testing.B) {
			cfg := CPUOnlyConfig()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithStrategy(a, mat, StrategyCacheBlocked, cfg)
			}
		})

		// Benchmark Accelerate (fp64)
		b.Run(fmt.Sprintf("Accelerate_FP64_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := backend.MatMul(a, mat)
				if err != nil {
					b.Fatalf("Accelerate MatMul failed: %v", err)
				}
			}
		})

		// Benchmark Accelerate (fp32)
		b.Run(fmt.Sprintf("Accelerate_FP32_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := backend.MatMulFloat32(a, mat)
				if err != nil {
					b.Fatalf("Accelerate MatMulFloat32 failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkAccelerateOverhead measures the overhead of Accelerate for small matrices.
func BenchmarkAccelerateOverhead(b *testing.B) {
	backend, err := NewAccelerateBackend()
	if err != nil {
		b.Skipf("Accelerate not available: %v", err)
	}

	// Small matrices where overhead matters
	sizes := []int{4, 8, 16, 32, 64}

	for _, size := range sizes {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := backend.MatMul(a, mat)
				if err != nil {
					b.Fatalf("Accelerate MatMul failed: %v", err)
				}
			}
		})
	}
}

// TestAccelerateErrorHandling tests error conditions.
func TestAccelerateErrorHandling(t *testing.T) {
	backend, err := NewAccelerateBackend()
	if err != nil {
		t.Fatalf("Failed to create Accelerate backend: %v", err)
	}

	// Test dimension mismatch
	a := NewTensor(4, 5)
	b := NewTensor(3, 6) // Incompatible

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
