package main

import (
	"runtime"
	"sync"
)

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file implements parallel tensor operations using Go's goroutines and
// channels. This demonstrates how to leverage Go's concurrency primitives to
// speed up compute-intensive operations like matrix multiplication.
//
// INTENTION:
// Show the performance progression from naive single-threaded → parallel
// multi-threaded operations. This is purely educational - demonstrating that
// concurrency can provide 2-8x speedups on multi-core machines, but is still
// memory-bandwidth limited for large matrices.
//
// GO CONCURRENCY PATTERNS DEMONSTRATED:
//
// 1. **Worker Pool Pattern**: Fixed number of goroutines processing work items
//    - Avoids goroutine creation overhead
//    - Provides predictable resource usage
//    - Good for batch processing
//
// 2. **Fan-Out/Fan-In Pattern**: Distribute work across goroutines, collect results
//    - Scales naturally with runtime.NumCPU()
//    - Uses channels for synchronization
//    - WaitGroup for completion tracking
//
// 3. **Row-Parallel MatMul**: Each goroutine computes a subset of output rows
//    - Simple work distribution (no shared state)
//    - Cache-friendly (each goroutine works on contiguous memory)
//    - Scales linearly up to memory bandwidth limit
//
// PERFORMANCE CHARACTERISTICS:
//
// Sequential MatMul (tensor.go):
//   - 512×512: ~10 ms (naive O(n³))
//   - 1024×1024: ~80 ms
//   - Single-threaded, simple, correct
//
// Parallel MatMul (this file):
//   - 512×512: ~2-4 ms (2-4x speedup on 8-core)
//   - 1024×1024: ~15-25 ms (3-5x speedup)
//   - Diminishing returns due to memory bandwidth
//
// Why not 8x speedup on 8 cores?
//   - Memory bandwidth bottleneck (loading A and B from RAM)
//   - Cache contention (cores competing for L3 cache)
//   - Synchronization overhead (WaitGroup, channels)
//
// Next optimizations (future work):
//   - Cache blocking/tiling: improves locality
//   - SIMD vectorization: processes multiple elements per instruction
//   - GPU offload: 50-200x for large matrices
//
// EDUCATIONAL VALUE:
// This file teaches:
//   1. How to parallelize compute-intensive algorithms in Go
//   2. Trade-offs between goroutines, threads, and synchronization
//   3. Why memory bandwidth matters more than CPU cores
//   4. How to measure and benchmark parallel performance
//
// ===========================================================================

// MatMulParallel performs parallel matrix multiplication: C = A × B.
// Splits work across goroutines, with each goroutine computing a subset of output rows.
//
// This uses the "row-parallel" strategy:
//   - Divide output matrix rows among numWorkers goroutines
//   - Each goroutine computes its assigned rows independently
//   - No shared state between goroutines (only read-only input matrices)
//   - Synchronize completion with sync.WaitGroup
//
// Parameters:
//   - a: (M, K) matrix
//   - b: (K, N) matrix
//   - numWorkers: number of goroutines (0 = runtime.NumCPU())
//
// Returns:
//   - (M, N) matrix C = A × B
//
// Performance:
//   - Best for M >> numWorkers (many rows to distribute)
//   - Limited by memory bandwidth for large matrices
//   - Typical speedup: 2-5x on 8-core machines
func MatMulParallel(a, b *Tensor, numWorkers int) *Tensor {
	// Validate shapes
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("tensor: MatMulParallel requires 2D tensors")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k1 != k2 {
		panic("tensor: incompatible dimensions for matmul")
	}
	k := k1

	// Auto-detect number of workers
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	// For small matrices, parallel overhead isn't worth it
	// Threshold: if M < numWorkers*2, use sequential
	if m < numWorkers*2 {
		return MatMul(a, b) // Fall back to sequential
	}

	out := NewTensor(m, n)

	// Compute rows per worker (distribute work evenly)
	rowsPerWorker := m / numWorkers
	remainder := m % numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	// Launch worker goroutines
	for w := 0; w < numWorkers; w++ {
		// Calculate this worker's row range
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker

		// Last worker takes remainder rows
		if w == numWorkers-1 {
			endRow += remainder
		}

		// Launch goroutine to compute rows [startRow, endRow)
		go func(start, end int) {
			defer wg.Done()

			// Each goroutine computes a subset of output rows
			// This is the SAME computation as the naive version,
			// just split across multiple goroutines
			for i := start; i < end; i++ {
				for j := 0; j < n; j++ {
					sum := 0.0
					for kk := 0; kk < k; kk++ {
						// Read from A[i, kk] and B[kk, j]
						// Multiple goroutines may read the same memory
						// (B is read by all workers), but no writes conflict
						sum += a.At(i, kk) * b.At(kk, j)
					}
					out.Set(sum, i, j)
				}
			}
		}(startRow, endRow)
	}

	// Wait for all workers to complete
	wg.Wait()

	return out
}

// MatMulParallelChannel is an alternative implementation using channels
// instead of pre-calculated row ranges. This demonstrates the "work queue"
// pattern where workers pull tasks from a shared channel.
//
// This version is slightly slower due to channel overhead, but demonstrates
// an important Go concurrency pattern. Use MatMulParallel for production.
//
// Educational value:
//   - Shows work queue pattern
//   - Demonstrates channel-based synchronization
//   - Highlights overhead of channel communication
func MatMulParallelChannel(a, b *Tensor, numWorkers int) *Tensor {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("tensor: MatMulParallelChannel requires 2D tensors")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k1 != k2 {
		panic("tensor: incompatible dimensions for matmul")
	}
	k := k1

	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	if m < numWorkers*2 {
		return MatMul(a, b)
	}

	out := NewTensor(m, n)

	// Create work queue: each task is a row index
	rowChan := make(chan int, m)
	for i := 0; i < m; i++ {
		rowChan <- i
	}
	close(rowChan) // Signal no more work

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	// Launch workers
	for w := 0; w < numWorkers; w++ {
		go func() {
			defer wg.Done()

			// Pull row indices from channel until empty
			for i := range rowChan {
				for j := 0; j < n; j++ {
					sum := 0.0
					for kk := 0; kk < k; kk++ {
						sum += a.At(i, kk) * b.At(kk, j)
					}
					out.Set(sum, i, j)
				}
			}
		}()
	}

	wg.Wait()
	return out
}

// ===========================================================================
// BENCHMARKING & PERFORMANCE ANALYSIS
// ===========================================================================
//
// To understand parallel performance, run:
//
//   go test -bench BenchmarkMatMul -benchtime=3s
//
// Expected results (8-core M1/M2 Mac):
//
//   BenchmarkMatMulSequential_512x512    200    10 ms/op
//   BenchmarkMatMulParallel_512x512      500     3 ms/op    (3x speedup)
//   BenchmarkMatMulSequential_1024x1024   15    80 ms/op
//   BenchmarkMatMulParallel_1024x1024     60    20 ms/op    (4x speedup)
//
// Why not 8x speedup?
//   1. Memory bandwidth: ~50 GB/s shared across all cores
//   2. Cache contention: L3 cache shared, causes stalls
//   3. Synchronization: WaitGroup adds ~100 ns overhead
//
// Amdahl's Law:
//   If 90% of work is parallelizable, max speedup = 10x (not 8x)
//   In practice: 3-5x is typical for memory-bound operations
//
// Next steps:
//   - Cache blocking (tensor_cache.go): 2-4x additional gain
//   - SIMD (tensor_simd.go): 2-4x on top of that
//   - GPU (metal_matmul.go): 50-200x for large matrices
//
// ===========================================================================
