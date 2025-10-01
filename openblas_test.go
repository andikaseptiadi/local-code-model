//go:build linux && arm64

package main

import (
	"math"
	"testing"
)

// TestOpenBLASAvailability tests OpenBLAS availability
func TestOpenBLASAvailability(t *testing.T) {
	backend, err := NewOpenBLASBackend()

	if err != nil {
		t.Logf("⚠️  OpenBLAS not available: %v", err)
		t.Logf("To install: sudo apt-get install libopenblas-dev")
		t.Skip("OpenBLAS not installed")
	}

	if backend == nil {
		t.Fatal("OpenBLAS backend should exist")
	}

	if !backend.IsAvailable() {
		t.Skip("OpenBLAS reports as unavailable")
	}

	t.Logf("✅ OpenBLAS available: %s", backend.DeviceName())
	t.Logf("Info: %s", backend.GetInfo())
}

// TestOpenBLASMatMulCorrectness tests OpenBLAS correctness
func TestOpenBLASMatMulCorrectness(t *testing.T) {
	backend, err := NewOpenBLASBackend()
	if err != nil || !backend.IsAvailable() {
		t.Skip("OpenBLAS not available")
	}

	sizes := []int{4, 8, 16, 32, 64}

	for _, size := range sizes {
		t.Run(string(rune('0'+size/10))+string(rune('0'+size%10)), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// OpenBLAS result
			result, err := backend.MatMul(a, b)
			if err != nil {
				t.Fatalf("OpenBLAS MatMul failed: %v", err)
			}

			// Reference result
			expected := MatMul(a, b)

			// Compare
			maxDiff := 0.0
			for i := range result.data {
				diff := math.Abs(result.data[i] - expected.data[i])
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			// OpenBLAS should be very accurate (within floating point precision)
			if maxDiff > 1e-10 {
				t.Errorf("OpenBLAS result differs: max diff %e", maxDiff)
			}

			t.Logf("Size %d: max diff %e ✅", size, maxDiff)
		})
	}
}

// TestNEONCorrectness tests NEON C intrinsics correctness
func TestNEONCorrectness(t *testing.T) {
	if !IsSIMDAvailable() {
		t.Skip("NEON not available (CGo disabled?)")
	}

	sizes := []int{4, 8, 16, 32, 64}

	for _, size := range sizes {
		t.Run(string(rune('0'+size/10))+string(rune('0'+size%10)), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// NEON result
			result := MatMulSIMD(a, b)

			// Reference result
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
				t.Errorf("NEON result differs: max diff %e", maxDiff)
			}

			t.Logf("Size %d: max diff %e ✅", size, maxDiff)
		})
	}
}

// BenchmarkOpenBLAS benchmarks OpenBLAS performance
func BenchmarkOpenBLAS(b *testing.B) {
	backend, err := NewOpenBLASBackend()
	if err != nil || !backend.IsAvailable() {
		b.Skip("OpenBLAS not available")
	}

	sizes := []int{128, 256, 512, 1024}

	for _, size := range sizes {
		b.Run(string(rune('0'+size/100))+string(rune('0'+(size/10)%10))+string(rune('0'+size%10)), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}
		})
	}
}

// BenchmarkComparison compares all implementations
func BenchmarkComparison(b *testing.B) {
	size := 512

	a := NewTensorRand(size, size)
	mat := NewTensorRand(size, size)

	// Naive
	b.Run("Naive", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = MatMul(a, mat)
		}
	})

	// NEON
	if IsSIMDAvailable() {
		b.Run("NEON", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = MatMulSIMD(a, mat)
			}
		})
	}

	// SVE
	sve, err := NewSVEBackend()
	if err == nil && sve.IsAvailable() {
		b.Run("SVE", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = sve.MatMul(a, mat)
			}
		})
	}

	// OpenBLAS
	openblas, err := NewOpenBLASBackend()
	if err == nil && openblas.IsAvailable() {
		b.Run("OpenBLAS", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = openblas.MatMul(a, mat)
			}
		})
	}
}
