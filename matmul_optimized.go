package main

import (
	"sync"
)

// RECOMMENDED READING:
//
// Cache Optimization:
// - "Computer Architecture: A Quantitative Approach" by Hennessy & Patterson
//   Chapter 2: Memory Hierarchy Design
//
// - "What Every Programmer Should Know About Memory" by Ulrich Drepper
//   https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
//
// SIMD:
// - "ARM NEON Programming Guide"
//   https://developer.arm.com/documentation/den0018/a/
//
// - "Optimizing Software in C++" by Agner Fog (applies to Go assembly)
//   https://www.agner.org/optimize/

// MatMul Optimization Levels:
// - Level 0: Naive triple loop (baseline)
// - Level 1: Parallel (goroutines)
// - Level 2: Cache-blocked (tiled)
// - Level 3: SIMD vectorized (assembly)
// - Level 4: Metal GPU
// - Level 5: ANE

// MatMulStrategy represents different matrix multiplication implementations.
type MatMulStrategy int

const (
	StrategyNaive MatMulStrategy = iota
	StrategyParallel
	StrategyCacheBlocked
	StrategySIMD
	StrategyMetal
	StrategyANE
)

// MatMulCacheBlocked performs cache-optimized matrix multiplication.
//
// CACHE HIERARCHY (M4 Max):
// - L1: 192 KB per P-core (64 KB instruction, 128 KB data)
// - L2: 16 MB shared across P-cores
// - L3: None (unified memory instead)
// - Main Memory: 400 GB/s bandwidth
//
// BLOCKING STRATEGY:
// Block size chosen to fit in L1 cache: 64x64 float64 = 32 KB
// Three blocks (A_block, B_block, C_block) = 96 KB < 128 KB L1 data cache
//
// PERFORMANCE GAIN:
// - Reduces cache misses from O(n³) to O(n³/B) where B = block size
// - Improves temporal locality (reuse of loaded data)
// - Expected speedup: 2-4x over naive for n > 256
func MatMulCacheBlocked(a, b *Tensor, blockSize int) *Tensor {
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

	// Block size tuned for M4 Max L1 cache
	// 64x64 float64 blocks = 32 KB per block
	// 3 active blocks < 128 KB L1 data cache
	if blockSize == 0 {
		blockSize = 64 // Default: tuned for M4 Max
	}

	// Blocked matrix multiplication
	// Outer loops tile the output matrix into blocks
	for i0 := 0; i0 < m; i0 += blockSize {
		iMax := min(i0+blockSize, m)

		for j0 := 0; j0 < n; j0 += blockSize {
			jMax := min(j0+blockSize, n)

			for k0 := 0; k0 < k; k0 += blockSize {
				kMax := min(k0+blockSize, k)

				// Inner loops work on a single block
				// This block fits in L1 cache
				for i := i0; i < iMax; i++ {
					for j := j0; j < jMax; j++ {
						sum := out.At(i, j) // Accumulate into existing value

						// Innermost loop: vector dot product
						// High cache hit rate - data is in L1
						for kk := k0; kk < kMax; kk++ {
							sum += a.At(i, kk) * b.At(kk, j)
						}

						out.Set(sum, i, j)
					}
				}
			}
		}
	}

	return out
}

// MatMulCacheBlockedParallel combines cache blocking with parallelism.
//
// STRATEGY:
// - Parallelize over output block rows (coarse-grained parallelism)
// - Each worker processes entire block rows to maximize cache locality
// - Minimal synchronization overhead
//
// PERFORMANCE:
// - Expected: 8-12x over naive on M4 Max (12 P-cores)
// - Cache blocking provides 2-4x
// - Parallelism provides additional 3-4x (not full 12x due to memory bandwidth)
func MatMulCacheBlockedParallel(a, b *Tensor, blockSize int, cfg ComputeConfig) *Tensor {
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

	if blockSize == 0 {
		blockSize = 64
	}

	if !cfg.shouldParallelize(m) {
		return MatMulCacheBlocked(a, b, blockSize)
	}

	// Calculate number of block rows
	numBlockRows := (m + blockSize - 1) / blockSize
	numWorkers := cfg.numWorkers()

	// Assign block rows to workers
	blockRowsPerWorker := (numBlockRows + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		startBlockRow := w * blockRowsPerWorker
		endBlockRow := min(startBlockRow+blockRowsPerWorker, numBlockRows)

		if startBlockRow >= numBlockRows {
			wg.Done()
			continue
		}

		go func(startBR, endBR int) {
			defer wg.Done()

			// Each worker processes a range of block rows
			for br := startBR; br < endBR; br++ {
				i0 := br * blockSize
				iMax := min(i0+blockSize, m)

				// Process this block row
				for j0 := 0; j0 < n; j0 += blockSize {
					jMax := min(j0+blockSize, n)

					for k0 := 0; k0 < k; k0 += blockSize {
						kMax := min(k0+blockSize, k)

						for i := i0; i < iMax; i++ {
							for j := j0; j < jMax; j++ {
								sum := out.At(i, j)
								for kk := k0; kk < kMax; kk++ {
									sum += a.At(i, kk) * b.At(kk, j)
								}
								out.Set(sum, i, j)
							}
						}
					}
				}
			}
		}(startBlockRow, endBlockRow)
	}

	wg.Wait()
	return out
}

// MatMulSIMD performs SIMD-vectorized matrix multiplication.
//
// ARM NEON (M4 Max):
// - 128-bit SIMD registers (4x float32 or 2x float64)
// - SVE (Scalable Vector Extension): up to 512-bit vectors
// - Operations: FADD, FMUL, FMA (fused multiply-add)
//
// VECTORIZATION STRATEGY:
// - Inner loop processes 4 elements at a time (float32)
// - Use FMA instructions for efficiency
// - Requires Go assembly or compiler auto-vectorization
//
// For now, this is a placeholder that shows the interface.
// Full implementation would use Go assembly (*.s files) with NEON intrinsics.
func MatMulSIMD(a, b *Tensor) *Tensor {
	// TODO: Implement with Go assembly using ARM NEON
	// For reference, see: src/math/dim_arm64.s in Go stdlib
	//
	// Expected implementation:
	// - Inner loop in assembly using NEON VLD/VST, FMLA
	// - Process 4 float32s or 2 float64s per iteration
	// - Requires block layout for efficient memory access

	// For now, fall back to cache-blocked version
	return MatMulCacheBlocked(a, b, 64)
}

// MatMulWithStrategy performs matrix multiplication using specified strategy.
func MatMulWithStrategy(a, b *Tensor, strategy MatMulStrategy, cfg BackendConfig) *Tensor {
	switch strategy {
	case StrategyNaive:
		return matmulSingleThreaded(a, b, NewTensor(a.shape[0], b.shape[1]),
			a.shape[0], b.shape[1], a.shape[1])

	case StrategyParallel:
		return ParallelMatMul(a, b, cfg.ComputeConfig)

	case StrategyCacheBlocked:
		return MatMulCacheBlocked(a, b, 64)

	case StrategySIMD:
		return MatMulSIMD(a, b)

	case StrategyMetal:
		// Try Metal, fall back to cache-blocked
		if metal, err := NewMetalBackend(); err == nil {
			if result, err := metal.MatMul(a, b); err == nil {
				return result
			}
		}
		return MatMulCacheBlocked(a, b, 64)

	case StrategyANE:
		// Try ANE, fall back to Metal, then cache-blocked
		if ane, err := NewANEBackend(); err == nil {
			if result, err := ane.MatMul(a, b); err == nil {
				return result
			}
		}
		// Fall through to Metal
		if metal, err := NewMetalBackend(); err == nil {
			if result, err := metal.MatMul(a, b); err == nil {
				return result
			}
		}
		return MatMulCacheBlocked(a, b, 64)

	default:
		panic("unknown strategy")
	}
}

// OptimizationBenchmark compares all optimization levels.
type OptimizationBenchmark struct {
	Strategy MatMulStrategy
	Name     string
	TimeNs   int64
	Speedup  float64
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}