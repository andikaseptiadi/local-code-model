// +build arm64,!cgo

package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file provides Go wrappers for SIMD-optimized matrix multiplication
// using ARM NEON instructions.
//
// INTENTION:
// Demonstrate instruction-level parallelism (ILP) via SIMD vectorization.
// Process multiple elements per CPU instruction for 2-4x speedup.
//
// WHEN TO USE THIS:
//
// 1. Learning: Understanding low-level optimization
// 2. Medium matrices (128-512): Less overhead than BLAS
// 3. Custom operations: Not available in standard libraries
// 4. Pure Go deployment: No CGo/framework dependencies
//
// WHEN NOT TO USE THIS:
//
// 1. Large matrices (>1024): Use Accelerate or Metal instead
// 2. Production code: BLAS is more mature and tested
// 3. Non-ARM platforms: This is ARM64-specific
//
// PERFORMANCE EXPECTATION:
//
// Compared to cache-blocked (~10 GFLOPS):
//   - Expected: 2-4x improvement (~20-40 GFLOPS)
//   - Reality: Depends on compiler, memory bandwidth, data size
//
// WHY NOT ALWAYS FASTER?
//
// SIMD helps with computation but not memory bandwidth:
//   - Small matrices: Memory-bound (SIMD helps less)
//   - Large matrices: Cache thrashing (need better tiling)
//   - Odd dimensions: Scalar cleanup reduces benefit
//
// IMPLEMENTATION NOTES:
//
// The assembly function expects:
//   - Row-major layout (C/Go convention)
//   - Contiguous memory (no striding)
//   - Compatible dimensions (a.cols == b.rows)
//
// ===========================================================================

// matmulNEON is implemented in matmul_neon_arm64.s
// It performs C = A @ B using ARM NEON SIMD instructions.
//
// Arguments:
//   a: matrix A in row-major layout (m x k)
//   b: matrix B in row-major layout (k x n)
//   c: output matrix C (m x n) - pre-allocated
//   m: number of rows in A
//   n: number of columns in B
//   k: number of columns in A (and rows in B)
func matmulNEON(a, b, c []float64, m, n, k int)

// MatMulSIMD performs matrix multiplication using SIMD vectorization.
//
// This is the Go wrapper that validates inputs and calls the assembly
// implementation.
//
// Performance characteristics:
//   - Best for medium matrices (128-512)
//   - 2-4x faster than cache-blocked
//   - Processes 2 float64 per instruction
//   - Lower overhead than Accelerate for small matrices
func MatMulSIMD(a, b *Tensor) *Tensor {
	// Validate inputs
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("MatMulSIMD: requires 2D tensors")
	}

	m := a.shape[0] // rows of A
	k := a.shape[1] // cols of A
	n := b.shape[1] // cols of B

	if b.shape[0] != k {
		panic("MatMulSIMD: incompatible dimensions")
	}

	// Allocate output
	c := NewTensor(m, n)

	// Call assembly implementation
	matmulNEON(a.data, b.data, c.data, m, n, k)

	return c
}

// IsSIMDAvailable checks if SIMD (NEON) is available on this platform.
//
// On ARM64, NEON is always available (it's part of the base ISA).
// This function exists for API consistency with other backends.
func IsSIMDAvailable() bool {
	return true // NEON is mandatory on ARM64
}

// SIMDInfo returns information about the SIMD capabilities.
func SIMDInfo() string {
	return "ARM NEON (128-bit vectors, 2x float64 per instruction)"
}
