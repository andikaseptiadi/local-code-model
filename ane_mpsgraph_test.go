package main

import (
	"fmt"
	"testing"
)

// ===========================================================================
// ANE via MPSGraph Tests
// ===========================================================================

// TestANEMPSGraphBasic tests basic ANE/MPSGraph functionality.
func TestANEMPSGraphBasic(t *testing.T) {
	// Create small executor for testing
	backend, err := NewANEBackendWithSize(4, 4, 4)
	if err != nil {
		t.Skipf("ANE/MPSGraph not available: %v", err)
	}
	defer backend.Close()

	// Create test matrices
	a := NewTensor(4, 4)
	b := NewTensor(4, 4)

	// Simple test: A = identity, B = ones
	for i := 0; i < 4; i++ {
		a.Set(1.0, i, i)
		for j := 0; j < 4; j++ {
			b.Set(1.0, i, j)
		}
	}

	// Execute
	result, err := backend.MatMul(a, b)
	if err != nil {
		t.Fatalf("MPSGraph MatMul failed: %v", err)
	}

	// Verify shape
	if result.shape[0] != 4 || result.shape[1] != 4 {
		t.Errorf("Result shape incorrect: got %v, want [4 4]", result.shape)
	}

	// Verify result: identity * ones = ones
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			expected := 1.0
			actual := result.At(i, j)
			if abs(actual-expected) > 1e-5 {
				t.Errorf("Result[%d,%d] = %f, want %f", i, j, actual, expected)
			}
		}
	}

	t.Logf("ANE/MPSGraph basic test passed")
}

// TestANEMPSGraphCorrectness compares ANE with CPU ground truth.
func TestANEMPSGraphCorrectness(t *testing.T) {
	size := 16
	backend, err := NewANEBackendWithSize(size, size, size)
	if err != nil {
		t.Skipf("ANE/MPSGraph not available: %v", err)
	}
	defer backend.Close()

	// Create random test matrices
	a := NewTensorRand(size, size)
	b := NewTensorRand(size, size)

	// CPU ground truth (using Accelerate for speed)
	accelerate, err := NewAccelerateBackend()
	if err != nil {
		t.Skipf("Accelerate not available: %v", err)
	}

	expected, err := accelerate.MatMul(a, b)
	if err != nil {
		t.Fatalf("Accelerate MatMul failed: %v", err)
	}

	// ANE result
	result, err := backend.MatMul(a, b)
	if err != nil {
		t.Fatalf("ANE MatMul failed: %v", err)
	}

	// Compare with tolerance for FP32 conversion
	maxDiff := 0.0
	for i := range result.data {
		diff := abs(result.data[i] - expected.data[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	// Tolerance for FP64->FP32->FP64 round-trip
	if maxDiff > 1e-4 {
		t.Errorf("ANE result differs from CPU: max diff %e (want < 1e-4)", maxDiff)
	}

	t.Logf("ANE correctness test passed (max diff: %e)", maxDiff)
}

// TestANEMPSGraphSizes tests different matrix sizes.
func TestANEMPSGraphSizes(t *testing.T) {
	sizes := []int{4, 8, 16, 32, 64}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			backend, err := NewANEBackendWithSize(size, size, size)
			if err != nil {
				t.Skipf("ANE/MPSGraph not available: %v", err)
			}
	
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			result, err := backend.MatMul(a, b)
			if err != nil {
				t.Fatalf("ANE MatMul failed for size %d: %v", size, err)
			}

			if result.shape[0] != size || result.shape[1] != size {
				t.Errorf("Result shape incorrect: got %v, want [%d %d]", result.shape, size, size)
			}
		})
	}
}

// BenchmarkANEMPSGraph benchmarks ANE/MPSGraph performance.
func BenchmarkANEMPSGraph(b *testing.B) {
	sizes := []int{128, 256, 512}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("ANE_%d", size), func(b *testing.B) {
			backend, err := NewANEBackendWithSize(size, size, size)
			if err != nil {
				b.Skipf("ANE/MPSGraph not available: %v", err)
			}
	
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := backend.MatMul(a, mat)
				if err != nil {
					b.Fatalf("ANE MatMul failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkANEvsMetalvsAccelerate compares all three.
func BenchmarkANEvsMetalvsAccelerate(b *testing.B) {
	size := 512

	// ANE
	b.Run("ANE", func(b *testing.B) {
		backend, err := NewANEBackendWithSize(size, size, size)
		if err != nil {
			b.Skipf("ANE not available: %v", err)
		}

		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = backend.MatMul(a, mat)
		}
	})

	// Metal
	b.Run("Metal", func(b *testing.B) {
		backend, err := NewMetalBackend()
		if err != nil {
			b.Skipf("Metal not available: %v", err)
		}

		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = backend.MatMul(a, mat)
		}
	})

	// Accelerate
	b.Run("Accelerate", func(b *testing.B) {
		backend, err := NewAccelerateBackend()
		if err != nil {
			b.Skipf("Accelerate not available: %v", err)
		}

		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = backend.MatMul(a, mat)
		}
	})
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
