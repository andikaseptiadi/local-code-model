package main

import (
	"fmt"
	"math"
	"runtime"
	"testing"
	"time"
)

func TestComputeConfig(t *testing.T) {
	// Test default config
	cfg := DefaultComputeConfig()
	if !cfg.Parallel {
		t.Error("default config should enable parallel execution")
	}
	if cfg.numWorkers() != runtime.NumCPU() {
		t.Errorf("expected %d workers, got %d", runtime.NumCPU(), cfg.numWorkers())
	}

	// Test single-threaded config
	stCfg := SingleThreadedConfig()
	if stCfg.Parallel {
		t.Error("single-threaded config should disable parallel execution")
	}
	if stCfg.numWorkers() != 1 {
		t.Errorf("single-threaded config should have 1 worker, got %d", stCfg.numWorkers())
	}
}

func TestParallelMatMulCorrectness(t *testing.T) {
	// Test that parallel MatMul produces same results as single-threaded
	sizes := []int{32, 64, 128, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// Compute with single-threaded
			stCfg := SingleThreadedConfig()
			resultST := MatMulWithConfig(a, b, stCfg)

			// Compute with parallel
			parCfg := DefaultComputeConfig()
			resultPar := MatMulWithConfig(a, b, parCfg)

			// Compare results
			if !tensorsEqual(resultST, resultPar, 1e-10) {
				t.Error("parallel and single-threaded results differ")
			}
		})
	}
}

func TestParallelMatMulPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping performance test in short mode")
	}

	size := 256
	iterations := 10

	a := NewTensorRand(size, size)
	b := NewTensorRand(size, size)

	// Benchmark single-threaded
	stCfg := SingleThreadedConfig()
	startST := time.Now()
	for i := 0; i < iterations; i++ {
		_ = MatMulWithConfig(a, b, stCfg)
	}
	durationST := time.Since(startST)

	// Benchmark parallel
	parCfg := DefaultComputeConfig()
	startPar := time.Now()
	for i := 0; i < iterations; i++ {
		_ = MatMulWithConfig(a, b, parCfg)
	}
	durationPar := time.Since(startPar)

	speedup := float64(durationST) / float64(durationPar)
	t.Logf("Size: %dx%d, Iterations: %d", size, size, iterations)
	t.Logf("Single-threaded: %v", durationST)
	t.Logf("Parallel:        %v", durationPar)
	t.Logf("Speedup:         %.2fx", speedup)
	t.Logf("Workers:         %d", runtime.NumCPU())

	// We expect at least some speedup (conservative check)
	if speedup < 1.5 {
		t.Logf("Warning: speedup is less than 1.5x, may indicate overhead or small problem size")
	}
}

func TestParallelApply(t *testing.T) {
	size := 10000
	x := NewTensorRand(size)

	fn := func(v float64) float64 {
		return v * 2.0
	}

	// Single-threaded
	stCfg := SingleThreadedConfig()
	resultST := ParallelApply(x, fn, stCfg)

	// Parallel
	parCfg := DefaultComputeConfig()
	resultPar := ParallelApply(x, fn, parCfg)

	// Compare
	if !tensorsEqual(resultST, resultPar, 1e-10) {
		t.Error("parallel and single-threaded apply differ")
	}
}

func TestMinSizeForParallel(t *testing.T) {
	// Test that small matrices use single-threaded path
	cfg := ComputeConfig{
		Parallel:           true,
		NumWorkers:         4,
		MinSizeForParallel: 100,
	}

	// Small matrix should use single-threaded
	if cfg.shouldParallelize(50) {
		t.Error("should not parallelize size 50 with threshold 100")
	}

	// Large matrix should use parallel
	if !cfg.shouldParallelize(200) {
		t.Error("should parallelize size 200 with threshold 100")
	}
}

func TestGlobalComputeConfig(t *testing.T) {
	// Save original config
	original := GetGlobalComputeConfig()
	defer SetGlobalComputeConfig(original)

	// Set single-threaded
	SetGlobalComputeConfig(SingleThreadedConfig())
	cfg := GetGlobalComputeConfig()
	if cfg.Parallel {
		t.Error("global config should be single-threaded")
	}

	// Set parallel
	SetGlobalComputeConfig(DefaultComputeConfig())
	cfg = GetGlobalComputeConfig()
	if !cfg.Parallel {
		t.Error("global config should be parallel")
	}
}

func TestComputeStats(t *testing.T) {
	stats := &ComputeStats{}

	stats.RecordOp(true, 1000)
	stats.RecordOp(false, 2000)
	stats.RecordOp(true, 1500)

	result := stats.GetStats()

	if result.TotalOps != 3 {
		t.Errorf("expected 3 total ops, got %d", result.TotalOps)
	}
	if result.ParallelOps != 2 {
		t.Errorf("expected 2 parallel ops, got %d", result.ParallelOps)
	}
	if result.SingleThreadedOps != 1 {
		t.Errorf("expected 1 single-threaded op, got %d", result.SingleThreadedOps)
	}
	if result.TotalTimeNs != 4500 {
		t.Errorf("expected 4500ns total time, got %d", result.TotalTimeNs)
	}

	stats.Reset()
	result = stats.GetStats()
	if result.TotalOps != 0 {
		t.Error("stats should be reset")
	}
}

// BenchmarkMatMulSingleThreaded benchmarks single-threaded matrix multiplication.
func BenchmarkMatMulSingleThreaded(b *testing.B) {
	sizes := []int{64, 128, 256}
	cfg := SingleThreadedConfig()

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithConfig(a, mat, cfg)
			}
		})
	}
}

// BenchmarkMatMulParallel benchmarks parallel matrix multiplication.
func BenchmarkMatMulParallel(b *testing.B) {
	sizes := []int{64, 128, 256}
	cfg := DefaultComputeConfig()

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithConfig(a, mat, cfg)
			}
		})
	}
}

// BenchmarkMatMulWorkerCounts benchmarks different worker counts.
func BenchmarkMatMulWorkerCounts(b *testing.B) {
	size := 256
	workerCounts := []int{1, 2, 4, 8, 12, 16}

	a := NewTensorRand(size, size)
	mat := NewTensorRand(size, size)

	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			cfg := ComputeConfig{
				Parallel:           workers > 1,
				NumWorkers:         workers,
				MinSizeForParallel: 64,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatMulWithConfig(a, mat, cfg)
			}
		})
	}
}

// Helper function to compare tensors with tolerance
func tensorsEqual(a, b *Tensor, tolerance float64) bool {
	if len(a.data) != len(b.data) {
		return false
	}

	for i := range a.data {
		if math.Abs(a.data[i]-b.data[i]) > tolerance {
			return false
		}
	}

	return true
}