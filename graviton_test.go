//go:build linux && arm64

package main

import (
	"fmt"
	"math"
	"testing"
)

// ===========================================================================
// Graviton-Specific Tests
// ===========================================================================

// TestGravitonGenerationDetection tests Graviton processor generation detection
func TestGravitonGenerationDetection(t *testing.T) {
	generation := GetGravitonGeneration()
	features := DetectCPUFeatures()
	cpuName := GetCPUName()

	t.Logf("CPU: %s", cpuName)
	t.Logf("Graviton Generation: %d", generation)
	t.Logf("NEON: %v, SVE: %v, SVE2: %v", features.HasNEON, features.HasSVE, features.HasSVE2)

	// Verify expected features for each Graviton generation
	switch generation {
	case 2:
		if !features.HasNEON {
			t.Error("Graviton2 should have NEON")
		}
		if features.HasSVE {
			t.Error("Graviton2 should not have SVE")
		}
		t.Log("✓ Graviton2: NEON only (expected)")

	case 3:
		if !features.HasNEON {
			t.Error("Graviton3 should have NEON")
		}
		if !features.HasSVE {
			t.Error("Graviton3 should have SVE")
		}
		if features.HasSVE2 {
			t.Error("Graviton3 should not have SVE2")
		}
		t.Log("✓ Graviton3: NEON + SVE (expected)")

	case 4:
		if !features.HasNEON {
			t.Error("Graviton4 should have NEON")
		}
		if !features.HasSVE {
			t.Error("Graviton4 should have SVE")
		}
		if !features.HasSVE2 {
			t.Error("Graviton4 should have SVE2")
		}
		t.Log("✓ Graviton4: NEON + SVE + SVE2 (expected)")

	default:
		t.Log("Not running on AWS Graviton")
	}
}

// TestSVEAvailability tests SVE backend availability
func TestSVEAvailability(t *testing.T) {
	backend, err := NewSVEBackend()

	generation := GetGravitonGeneration()

	if generation >= 3 {
		// Graviton3+ should have SVE
		if err != nil {
			t.Errorf("SVE should be available on Graviton3+: %v", err)
		}
		if !backend.IsAvailable() {
			t.Error("SVE backend should report as available")
		}

		vl := backend.VectorLength()
		t.Logf("SVE Vector Length: %d float64 elements", vl)

		// Expected vector lengths
		expectedVL := 4 // 256-bit / 64-bit = 4 elements on Graviton3
		if generation == 4 {
			expectedVL = 8 // Could be up to 512-bit / 64-bit = 8 elements
		}

		if vl < expectedVL {
			t.Logf("Warning: Expected at least %d elements, got %d", expectedVL, vl)
		}

	} else {
		// Graviton2 or non-Graviton should not have SVE
		if err == nil {
			t.Error("SVE should not be available on Graviton2 or non-Graviton")
		}
		t.Logf("SVE not available (expected): %v", err)
	}
}

// TestSVEMatMulCorrectness tests SVE matrix multiplication correctness
func TestSVEMatMulCorrectness(t *testing.T) {
	backend, err := NewSVEBackend()
	if err != nil {
		t.Skipf("SVE not available: %v", err)
	}

	sizes := []int{4, 8, 16, 32}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// SVE result
			result, err := backend.MatMul(a, b)
			if err != nil {
				t.Fatalf("SVE MatMul failed: %v", err)
			}

			// Naive result for comparison
			expected := MatMul(a, b)

			// Compare
			maxDiff := 0.0
			for i := range result.data {
				diff := math.Abs(result.data[i] - expected.data[i])
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			if maxDiff > 1e-10 {
				t.Errorf("SVE result differs from naive: max diff %e", maxDiff)
			}

			t.Logf("Size %d: max diff %e", size, maxDiff)
		})
	}
}

// BenchmarkGravitonNEON benchmarks NEON on Graviton
func BenchmarkGravitonNEON(b *testing.B) {
	if GetGravitonGeneration() == 0 {
		b.Skip("Not running on Graviton")
	}

	sizes := []int{128, 256, 512}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("NEON_%d", size), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulSIMD(a, mat)
			}
		})
	}
}

// BenchmarkGravitonSVE benchmarks SVE on Graviton3/4
func BenchmarkGravitonSVE(b *testing.B) {
	backend, err := NewSVEBackend()
	if err != nil {
		b.Skipf("SVE not available: %v", err)
	}

	sizes := []int{128, 256, 512}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("SVE_%d", size), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}
		})
	}
}

// BenchmarkGravitonComparison compares all backends on Graviton
func BenchmarkGravitonComparison(b *testing.B) {
	if GetGravitonGeneration() == 0 {
		b.Skip("Not running on Graviton")
	}

	size := 512

	// Naive
	b.Run("Naive", func(b *testing.B) {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = MatMul(a, mat)
		}
	})

	// SIMD (NEON on ARM)
	b.Run("SIMD", func(b *testing.B) {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = MatMulSIMD(a, mat)
		}
	})

	// SVE (if available)
	backend, err := NewSVEBackend()
	if err == nil {
		b.Run("SVE", func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}
		})
	}
}
