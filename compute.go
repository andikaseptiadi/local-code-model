package main

import (
	"runtime"
	"sync"
)

// ComputeConfig controls parallelization behavior for tensor operations.
//
// This allows switching between single-threaded (deterministic, easier debugging)
// and multi-threaded (faster) execution modes.
type ComputeConfig struct {
	// Parallel enables multi-threaded execution of tensor operations.
	Parallel bool

	// NumWorkers specifies the number of worker goroutines to use.
	// If 0, defaults to runtime.NumCPU().
	// Only used when Parallel is true.
	NumWorkers int

	// MinSizeForParallel specifies the minimum matrix dimension
	// before parallelization is used. Small matrices don't benefit
	// from parallelization due to goroutine overhead.
	MinSizeForParallel int
}

// DefaultComputeConfig returns a sensible default configuration.
func DefaultComputeConfig() ComputeConfig {
	return ComputeConfig{
		Parallel:           true,
		NumWorkers:         0, // Use all available CPUs
		MinSizeForParallel: 64,
	}
}

// SingleThreadedConfig returns a configuration for single-threaded execution.
func SingleThreadedConfig() ComputeConfig {
	return ComputeConfig{
		Parallel:           false,
		NumWorkers:         1,
		MinSizeForParallel: 0,
	}
}

// numWorkers returns the actual number of workers to use.
func (c ComputeConfig) numWorkers() int {
	if !c.Parallel {
		return 1
	}
	if c.NumWorkers > 0 {
		return c.NumWorkers
	}
	return runtime.NumCPU()
}

// shouldParallelize determines if an operation should use parallelization
// based on the problem size.
func (c ComputeConfig) shouldParallelize(size int) bool {
	return c.Parallel && size >= c.MinSizeForParallel
}

// Global compute configuration (can be overridden per operation)
var globalComputeConfig = DefaultComputeConfig()

// SetGlobalComputeConfig sets the global compute configuration.
func SetGlobalComputeConfig(cfg ComputeConfig) {
	globalComputeConfig = cfg
}

// GetGlobalComputeConfig returns the current global compute configuration.
func GetGlobalComputeConfig() ComputeConfig {
	return globalComputeConfig
}

// ParallelMatMul performs parallel matrix multiplication: C = A @ B.
//
// Parallelization strategy:
// - Divide output rows among workers
// - Each worker computes a contiguous block of rows
// - Minimizes false sharing (workers write to different cache lines)
//
// Performance characteristics:
// - Overhead: ~50-100µs for goroutine spawning and coordination
// - Speedup: Linear up to memory bandwidth limit (~8-10x on M4 Max)
// - Memory: No additional allocations beyond output matrix
func ParallelMatMul(a, b *Tensor, cfg ComputeConfig) *Tensor {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("tensor: MatMul requires 2D tensors")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]

	if k1 != k2 {
		panic("tensor: incompatible dimensions for matmul")
	}
	k := k1

	out := NewTensor(m, n)

	// Use single-threaded path for small matrices
	if !cfg.shouldParallelize(m) || !cfg.shouldParallelize(n) {
		return matmulSingleThreaded(a, b, out, m, n, k)
	}

	// Parallel execution
	numWorkers := cfg.numWorkers()
	rowsPerWorker := (m + numWorkers - 1) / numWorkers // Ceiling division

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > m {
			endRow = m
		}

		if startRow >= m {
			wg.Done()
			continue
		}

		go func(start, end int) {
			defer wg.Done()
			matmulWorker(a, b, out, start, end, n, k)
		}(startRow, endRow)
	}

	wg.Wait()
	return out
}

// matmulWorker computes a subset of output rows.
func matmulWorker(a, b, out *Tensor, startRow, endRow, n, k int) {
	for i := startRow; i < endRow; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			// Inner loop: dot product of row i from A with column j from B
			for kk := 0; kk < k; kk++ {
				sum += a.At(i, kk) * b.At(kk, j)
			}
			out.Set(sum, i, j)
		}
	}
}

// matmulSingleThreaded performs single-threaded matrix multiplication.
func matmulSingleThreaded(a, b, out *Tensor, m, n, k int) *Tensor {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for kk := 0; kk < k; kk++ {
				sum += a.At(i, kk) * b.At(kk, j)
			}
			out.Set(sum, i, j)
		}
	}
	return out
}

// ParallelApply applies a function to each element in parallel.
// Useful for element-wise operations like activations on large tensors.
func ParallelApply(t *Tensor, fn func(float64) float64, cfg ComputeConfig) *Tensor {
	out := NewTensor(t.shape...)
	size := len(t.data)

	if !cfg.shouldParallelize(size) {
		// Single-threaded
		for i := 0; i < size; i++ {
			out.data[i] = fn(t.data[i])
		}
		return out
	}

	// Parallel execution
	numWorkers := cfg.numWorkers()
	elemsPerWorker := (size + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		start := w * elemsPerWorker
		end := start + elemsPerWorker
		if end > size {
			end = size
		}

		if start >= size {
			wg.Done()
			continue
		}

		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				out.data[i] = fn(t.data[i])
			}
		}(start, end)
	}

	wg.Wait()
	return out
}

// MatMulWithConfig performs matrix multiplication with specified compute config.
func MatMulWithConfig(a, b *Tensor, cfg ComputeConfig) *Tensor {
	if cfg.Parallel {
		return ParallelMatMul(a, b, cfg)
	}
	return MatMul(a, b)
}

// ComputeStats tracks performance statistics for compute operations.
type ComputeStats struct {
	mu                sync.Mutex
	TotalOps          int64
	ParallelOps       int64
	SingleThreadedOps int64
	TotalTimeNs       int64
}

var globalStats ComputeStats

// RecordOp records a compute operation for statistics.
func (cs *ComputeStats) RecordOp(parallel bool, durationNs int64) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	cs.TotalOps++
	cs.TotalTimeNs += durationNs

	if parallel {
		cs.ParallelOps++
	} else {
		cs.SingleThreadedOps++
	}
}

// GetStats returns a copy of the current statistics.
func (cs *ComputeStats) GetStats() ComputeStats {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	return ComputeStats{
		TotalOps:          cs.TotalOps,
		ParallelOps:       cs.ParallelOps,
		SingleThreadedOps: cs.SingleThreadedOps,
		TotalTimeNs:       cs.TotalTimeNs,
	}
}

// Reset clears all statistics.
func (cs *ComputeStats) Reset() {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	cs.TotalOps = 0
	cs.ParallelOps = 0
	cs.SingleThreadedOps = 0
	cs.TotalTimeNs = 0
}