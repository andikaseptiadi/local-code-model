// +build arm64,!cgo

package main

import (
	"fmt"
	"math"
	"testing"
)

// ===========================================================================
// SIMD (NEON) Tests
// ===========================================================================
//
// These tests verify correctness and performance of the ARM NEON SIMD
// implementation.
//
// TEST STRATEGY:
//
// 1. Correctness: Compare SIMD results against naive CPU implementation
// 2. Edge cases: Small matrices, odd dimensions, single row/column
// 3. Performance: Benchmark against cache-blocked and Accelerate
//
// ===========================================================================

// TestSIMDAvailability checks if SIMD is available on this platform.
func TestSIMDAvailability(t *testing.T) {
	if !IsSIMDAvailable() {
		t.Skip("SIMD not available on this platform")
	}

	t.Logf("SIMD Info: %s", SIMDInfo())
}

// TestSIMDCorrectness validates SIMD results against naive implementation.
func TestSIMDCorrectness(t *testing.T) {
	if !IsSIMDAvailable() {
		t.Skip("SIMD not available")
	}

	sizes := []int{4, 8, 16, 32, 64, 128}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// CPU result (ground truth)
			cpuConfig := CPUOnlyConfig()
			cpuResult := MatMulWithStrategy(a, b, StrategyNaive, cpuConfig)

			// SIMD result
			simdResult := MatMulSIMD(a, b)

			// Compare with tight tolerance (both use float64)
			if !tensorsEqualApprox(cpuResult, simdResult, 1e-10) {
				t.Errorf("SIMD result differs from CPU")

				// Show first mismatch for debugging
				for i := 0; i < min(size, 4); i++ {
					for j := 0; j < min(size, 4); j++ {
						cpu := cpuResult.At(i, j)
						simd := simdResult.At(i, j)
						diff := math.Abs(cpu - simd)
						if diff > 1e-10 {
							t.Logf("First mismatch at [%d,%d]: CPU=%.15f, SIMD=%.15f, diff=%.15e",
								i, j, cpu, simd, diff)
							return
						}
					}
				}
			}
		})
	}
}

// TestSIMDEdgeCases tests edge cases like small matrices and odd dimensions.
func TestSIMDEdgeCases(t *testing.T) {
	if !IsSIMDAvailable() {
		t.Skip("SIMD not available")
	}

	testCases := []struct {
		name string
		m, n, k int
	}{
		{"1x1x1", 1, 1, 1},
		{"2x2x2", 2, 2, 2},
		{"3x3x3", 3, 3, 3}, // Odd dimension
		{"5x7x11", 5, 7, 11}, // Prime dimensions
		{"1x100x1", 1, 100, 1}, // Single row
		{"100x1x100", 100, 1, 100}, // Single column
		{"13x17x19", 13, 17, 19}, // All odd
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := NewTensorRand(tc.m, tc.k)
			b := NewTensorRand(tc.k, tc.n)

			// CPU result
			cpuConfig := CPUOnlyConfig()
			cpuResult := MatMulWithStrategy(a, b, StrategyNaive, cpuConfig)

			// SIMD result
			simdResult := MatMulSIMD(a, b)

			// Compare
			if !tensorsEqualApprox(cpuResult, simdResult, 1e-10) {
				t.Errorf("SIMD result differs from CPU for %s", tc.name)
			}
		})
	}
}

// TestSIMDVsCacheBlocked compares SIMD against cache-blocked implementation.
func TestSIMDVsCacheBlocked(t *testing.T) {
	if !IsSIMDAvailable() {
		t.Skip("SIMD not available")
	}

	sizes := []int{32, 64, 128, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// Cache-blocked result
			cachedResult := MatMulCacheBlocked(a, b, 64)

			// SIMD result
			simdResult := MatMulSIMD(a, b)

			// Compare (should be identical, both use float64)
			if !tensorsEqualApprox(cachedResult, simdResult, 1e-10) {
				t.Errorf("SIMD result differs from cache-blocked")

				// Show first mismatch
				for i := 0; i < min(size, 4); i++ {
					for j := 0; j < min(size, 4); j++ {
						cached := cachedResult.At(i, j)
						simd := simdResult.At(i, j)
						diff := math.Abs(cached - simd)
						if diff > 1e-10 {
							t.Logf("First mismatch at [%d,%d]: Cached=%.15f, SIMD=%.15f, diff=%.15e",
								i, j, cached, simd, diff)
							return
						}
					}
				}
			}
		})
	}
}

// TestSIMDNonSquare tests non-square matrices.
func TestSIMDNonSquare(t *testing.T) {
	if !IsSIMDAvailable() {
		t.Skip("SIMD not available")
	}

	testCases := []struct {
		name string
		m, n, k int
	}{
		{"tall", 100, 10, 50},
		{"wide", 10, 100, 50},
		{"tall_odd", 99, 11, 51},
		{"wide_odd", 11, 99, 51},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := NewTensorRand(tc.m, tc.k)
			b := NewTensorRand(tc.k, tc.n)

			// CPU result
			cpuConfig := CPUOnlyConfig()
			cpuResult := MatMulWithStrategy(a, b, StrategyNaive, cpuConfig)

			// SIMD result
			simdResult := MatMulSIMD(a, b)

			// Compare
			if !tensorsEqualApprox(cpuResult, simdResult, 1e-10) {
				t.Errorf("SIMD result differs from CPU for %s", tc.name)
			}
		})
	}
}

// BenchmarkSIMD benchmarks SIMD performance across different matrix sizes.
func BenchmarkSIMD(b *testing.B) {
	if !IsSIMDAvailable() {
		b.Skip("SIMD not available")
	}

	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		// Benchmark SIMD
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulSIMD(a, mat)
			}
		})

		// Benchmark cache-blocked for comparison
		b.Run(fmt.Sprintf("CacheBlocked_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulCacheBlocked(a, mat, 64)
			}
		})
	}
}

// BenchmarkSIMDVsAccelerate compares SIMD against Accelerate BLAS.
func BenchmarkSIMDVsAccelerate(b *testing.B) {
	if !IsSIMDAvailable() {
		b.Skip("SIMD not available")
	}

	// Check if Accelerate is available
	accel, err := NewAccelerateBackend()
	if err != nil {
		b.Skip("Accelerate not available")
	}

	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		// Benchmark SIMD
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulSIMD(a, mat)
			}
		})

		// Benchmark Accelerate
		b.Run(fmt.Sprintf("Accelerate_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = accel.MatMul(a, mat)
			}
		})
	}
}

// BenchmarkOptimizationProgression shows the full optimization continuum.
func BenchmarkOptimizationProgression(b *testing.B) {
	if !IsSIMDAvailable() {
		b.Skip("SIMD not available")
	}

	size := 256
	a := NewTensorRand(size, size)
	mat := NewTensorRand(size, size)

	strategies := []struct {
		name     string
		strategy MatMulStrategy
	}{
		{"1_Naive", StrategyNaive},
		{"2_Parallel", StrategyParallel},
		{"3_CacheBlocked", StrategyCacheBlocked},
		{"4_SIMD", StrategySIMD},
		{"5_Accelerate", StrategyAccelerate},
	}

	cfg := DefaultBackendConfig()

	for _, s := range strategies {
		b.Run(s.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithStrategy(a, mat, s.strategy, cfg)
			}
		})
	}
}

// TestSIMDWithStrategy tests SIMD via MatMulWithStrategy.
func TestSIMDWithStrategy(t *testing.T) {
	if !IsSIMDAvailable() {
		t.Skip("SIMD not available")
	}

	size := 64
	a := NewTensorRand(size, size)
	b := NewTensorRand(size, size)

	cfg := DefaultBackendConfig()

	// Get results from different strategies
	naive := MatMulWithStrategy(a, b, StrategyNaive, cfg)
	simd := MatMulWithStrategy(a, b, StrategySIMD, cfg)

	// Compare
	if !tensorsEqualApprox(naive, simd, 1e-10) {
		t.Error("SIMD via MatMulWithStrategy differs from naive")
	}
}
