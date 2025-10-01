package main

import (
	"fmt"
	"math"
	"testing"
)

func TestCacheBlockedCorrectness(t *testing.T) {
	sizes := []int{32, 64, 128, 256}
	blockSizes := []int{16, 32, 64}

	for _, size := range sizes {
		for _, blockSize := range blockSizes {
			t.Run(fmt.Sprintf("size=%d_block=%d", size, blockSize), func(t *testing.T) {
				a := NewTensorRand(size, size)
				b := NewTensorRand(size, size)

				// Naive result (ground truth)
				cfg := CPUOnlyConfig()
				resultNaive := MatMulWithStrategy(a, b, StrategyNaive, cfg)

				// Cache-blocked result
				resultBlocked := MatMulCacheBlocked(a, b, blockSize)

				// Compare
				if !tensorsEqualApprox(resultNaive, resultBlocked, 1e-8) {
					t.Error("cache-blocked result differs from naive")
				}
			})
		}
	}
}

func TestCacheBlockedParallelCorrectness(t *testing.T) {
	sizes := []int{64, 128, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// Naive result
			cfg := CPUOnlyConfig()
			resultNaive := MatMulWithStrategy(a, b, StrategyNaive, cfg)

			// Cache-blocked parallel
			cfgPar := DefaultComputeConfig()
			resultPar := MatMulCacheBlockedParallel(a, b, 64, cfgPar)

			if !tensorsEqualApprox(resultNaive, resultPar, 1e-8) {
				t.Error("cache-blocked parallel result differs from naive")
			}
		})
	}
}

func TestMatMulWithStrategy(t *testing.T) {
	size := 64
	a := NewTensorRand(size, size)
	b := NewTensorRand(size, size)

	cfg := CPUOnlyConfig()
	resultNaive := MatMulWithStrategy(a, b, StrategyNaive, cfg)

	strategies := []struct {
		name     string
		strategy MatMulStrategy
		cfg      BackendConfig
	}{
		{"Parallel", StrategyParallel, DefaultBackendConfig()},
		{"CacheBlocked", StrategyCacheBlocked, CPUOnlyConfig()},
		{"SIMD", StrategySIMD, CPUOnlyConfig()},
	}

	for _, s := range strategies {
		t.Run(s.name, func(t *testing.T) {
			result := MatMulWithStrategy(a, b, s.strategy, s.cfg)

			if !tensorsEqualApprox(resultNaive, result, 1e-8) {
				t.Errorf("%s result differs from naive", s.name)
			}
		})
	}
}

func BenchmarkMatMulStrategies(b *testing.B) {
	sizes := []int{128, 256, 512}

	for _, size := range sizes {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.Run(fmt.Sprintf("Naive_%d", size), func(b *testing.B) {
			cfg := CPUOnlyConfig()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithStrategy(a, mat, StrategyNaive, cfg)
			}
		})

		b.Run(fmt.Sprintf("Parallel_%d", size), func(b *testing.B) {
			cfg := DefaultBackendConfig()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithStrategy(a, mat, StrategyParallel, cfg)
			}
		})

		b.Run(fmt.Sprintf("CacheBlocked_%d", size), func(b *testing.B) {
			cfg := CPUOnlyConfig()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithStrategy(a, mat, StrategyCacheBlocked, cfg)
			}
		})

		b.Run(fmt.Sprintf("CachedParallel_%d", size), func(b *testing.B) {
			cfg := DefaultBackendConfig()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulCacheBlockedParallel(a, mat, 64, cfg.ComputeConfig)
			}
		})
	}
}

func BenchmarkCacheBlockSizes(b *testing.B) {
	size := 512
	a := NewTensorRand(size, size)
	mat := NewTensorRand(size, size)

	blockSizes := []int{16, 32, 64, 128, 256}

	for _, blockSize := range blockSizes {
		b.Run(fmt.Sprintf("block=%d", blockSize), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulCacheBlocked(a, mat, blockSize)
			}
		})
	}
}

// tensorsEqualApprox compares tensors with floating point tolerance
func tensorsEqualApprox(a, b *Tensor, tolerance float64) bool {
	if len(a.data) != len(b.data) {
		return false
	}

	for i := range a.data {
		diff := math.Abs(a.data[i] - b.data[i])
		if diff > tolerance {
			// Allow relative error for large values
			relError := diff / math.Max(math.Abs(a.data[i]), math.Abs(b.data[i]))
			if relError > tolerance {
				return false
			}
		}
	}

	return true
}