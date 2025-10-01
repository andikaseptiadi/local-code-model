// +build darwin,cgo

package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file provides bindings to Apple's Accelerate framework, specifically
// the BLAS (Basic Linear Algebra Subprograms) routines for matrix operations.
//
// INTENTION:
// Use Apple's highly optimized BLAS implementation which leverages:
//   - CPU SIMD instructions (NEON on ARM, AVX on Intel)
//   - Multi-threading for large matrices
//   - Cache-aware algorithms
//   - Hardware-specific optimizations
//
// WHY ACCELERATE VS METAL:
//
// Accelerate (BLAS):
//   + Easier to use (simple C API, no buffer management)
//   + Lower overhead (~microseconds vs milliseconds)
//   + Better for small-medium matrices (< 2048×2048)
//   + Automatically multi-threaded
//   + Works on both Intel and Apple Silicon
//   - Limited to CPU (doesn't use GPU)
//   - Peak: ~100-200 GFLOPS on M4 Max
//
// Metal (MPS):
//   + Uses GPU (massively parallel)
//   + Better for large matrices (> 2048×2048)
//   + Peak: ~4000 GFLOPS on M4 Max
//   - Higher overhead (1-5ms for data copy)
//   - More complex to use
//   - Only on Apple Silicon
//
// WHEN TO USE EACH:
//
// Small matrices (< 256×256):
//   → CPU cache-blocked (lowest overhead)
//
// Medium matrices (256×256 to 2048×2048):
//   → Accelerate BLAS (best performance/overhead tradeoff)
//
// Large matrices (> 2048×2048):
//   → Metal GPU (highest throughput)
//
// WHERE THIS SITS ON THE CONTINUUM:
//
// Level 3.5: Accelerate BLAS
//   - Between cache-blocked parallel (Level 3) and Metal (Level 4)
//   - Uses: CPU + SIMD + cache optimization + multi-threading
//   - Expected: 50-200 GFLOPS depending on matrix size
//   - Still stranded: GPU, ANE
//
// BLAS EXPLAINED:
//
// BLAS is a standard API for linear algebra with three levels:
//   - Level 1: Vector operations (O(n)) - dot products, norms
//   - Level 2: Matrix-vector (O(n²)) - matrix-vector multiply
//   - Level 3: Matrix-matrix (O(n³)) - matrix multiply (our focus)
//
// We use SGEMM (Single-precision GEneral Matrix-Matrix multiply):
//   C = alpha * op(A) * op(B) + beta * C
//
// Where:
//   - op(X) = X or X^T (transpose)
//   - alpha, beta = scalars
//   - For simple C = A @ B, use: alpha=1, beta=0, no transpose
//
// PERFORMANCE CHARACTERISTICS:
//
// Accelerate BLAS on M4 Max:
//   - Small (128×128): ~10 GFLOPS (overhead dominates)
//   - Medium (512×512): ~100 GFLOPS (sweet spot)
//   - Large (2048×2048): ~150 GFLOPS (memory bandwidth limited)
//
// Compare to:
//   - Naive: ~1 GFLOPS
//   - Cache-blocked: ~10 GFLOPS
//   - Cache-blocked + parallel: ~50 GFLOPS
//   - Metal GPU: ~500 GFLOPS (large matrices)
//
// ===========================================================================

/*
#cgo CFLAGS: -I/System/Library/Frameworks/Accelerate.framework/Headers -Wno-deprecated-declarations
#cgo LDFLAGS: -framework Accelerate

#include <Accelerate/Accelerate.h>

// Wrapper for SGEMM (single-precision matrix multiply)
// Computes: C = alpha * A @ B + beta * C
//
// Parameters match BLAS convention:
//   - Order: Row-major (CblasRowMajor) or column-major
//   - TransA/TransB: Transpose flags
//   - M, N, K: Matrix dimensions
//   - alpha, beta: Scalars
//   - A, B, C: Matrix data pointers
//   - lda, ldb, ldc: Leading dimensions (stride)
void accelerate_sgemm(
    int m, int n, int k,
    float alpha,
    const float* a, int lda,
    const float* b, int ldb,
    float beta,
    float* c, int ldc
) {
    // CBLAS: C interface to BLAS
    // Row-major order (C convention, not Fortran)
    cblas_sgemm(
        CblasRowMajor,      // Row-major layout (C/Go style)
        CblasNoTrans,       // Don't transpose A
        CblasNoTrans,       // Don't transpose B
        m, n, k,            // Matrix dimensions
        alpha,              // Scalar multiplier for A*B
        a, lda,             // Matrix A and its stride
        b, ldb,             // Matrix B and its stride
        beta,               // Scalar multiplier for C
        c, ldc              // Matrix C (output) and its stride
    );
}

// Wrapper for DGEMM (double-precision matrix multiply)
// Same as SGEMM but with float64
void accelerate_dgemm(
    int m, int n, int k,
    double alpha,
    const double* a, int lda,
    const double* b, int ldb,
    double beta,
    double* c, int ldc
) {
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc
    );
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// AccelerateBackend uses Apple's Accelerate framework for matrix operations.
type AccelerateBackend struct {
	available bool
}

// NewAccelerateBackend creates an Accelerate compute backend.
// Accelerate is always available on macOS (both Intel and Apple Silicon).
func NewAccelerateBackend() (*AccelerateBackend, error) {
	// Accelerate is built into macOS, always available
	return &AccelerateBackend{
		available: true,
	}, nil
}

// MatMul performs matrix multiplication using Accelerate BLAS.
//
// Uses DGEMM (double-precision) to match Go's float64 tensors.
// For fp32, would use SGEMM which is slightly faster.
//
// BLAS computes: C = alpha * A @ B + beta * C
// We use: alpha = 1.0, beta = 0.0 for simple C = A @ B
func (a *AccelerateBackend) MatMul(x, y *Tensor) (*Tensor, error) {
	if len(x.shape) != 2 || len(y.shape) != 2 {
		return nil, fmt.Errorf("accelerate: MatMul requires 2D tensors")
	}

	m := x.shape[0]  // Rows of A
	k := x.shape[1]  // Cols of A, Rows of B
	n := y.shape[1]  // Cols of B

	if x.shape[1] != y.shape[0] {
		return nil, fmt.Errorf("accelerate: incompatible dimensions for matmul: (%d,%d) @ (%d,%d)",
			x.shape[0], x.shape[1], y.shape[0], y.shape[1])
	}

	// Create output tensor
	out := NewTensor(m, n)

	// Call DGEMM via CGo
	// Parameters:
	//   m, n, k: Matrix dimensions
	//   alpha = 1.0: Multiply A*B by 1
	//   a, lda: Matrix A, leading dimension (stride) = k
	//   b, ldb: Matrix B, leading dimension = n
	//   beta = 0.0: Don't add to C (C = A*B, not C += A*B)
	//   c, ldc: Matrix C (output), leading dimension = n
	C.accelerate_dgemm(
		C.int(m),
		C.int(n),
		C.int(k),
		C.double(1.0), // alpha
		(*C.double)(unsafe.Pointer(&x.data[0])),
		C.int(k), // lda: leading dimension of A
		(*C.double)(unsafe.Pointer(&y.data[0])),
		C.int(n), // ldb: leading dimension of B
		C.double(0.0), // beta
		(*C.double)(unsafe.Pointer(&out.data[0])),
		C.int(n), // ldc: leading dimension of C
	)

	return out, nil
}

// IsAvailable returns true if Accelerate is available.
// Always true on macOS.
func (a *AccelerateBackend) IsAvailable() bool {
	return a.available
}

// Name returns the backend name.
func (a *AccelerateBackend) Name() string {
	return "Apple Accelerate (BLAS)"
}

// MatMulFloat32 performs matrix multiplication using single-precision SGEMM.
// Slightly faster than DGEMM but loses precision.
// Useful when fp32 accuracy is sufficient (most ML workloads).
func (a *AccelerateBackend) MatMulFloat32(x, y *Tensor) (*Tensor, error) {
	if len(x.shape) != 2 || len(y.shape) != 2 {
		return nil, fmt.Errorf("accelerate: MatMul requires 2D tensors")
	}

	m := x.shape[0]
	k := x.shape[1]
	n := y.shape[1]

	if x.shape[1] != y.shape[0] {
		return nil, fmt.Errorf("accelerate: incompatible dimensions")
	}

	// Convert to float32
	x32 := make([]float32, len(x.data))
	y32 := make([]float32, len(y.data))
	out32 := make([]float32, m*n)

	for i := range x.data {
		x32[i] = float32(x.data[i])
	}
	for i := range y.data {
		y32[i] = float32(y.data[i])
	}

	// Call SGEMM
	C.accelerate_sgemm(
		C.int(m),
		C.int(n),
		C.int(k),
		C.float(1.0),
		(*C.float)(unsafe.Pointer(&x32[0])),
		C.int(k),
		(*C.float)(unsafe.Pointer(&y32[0])),
		C.int(n),
		C.float(0.0),
		(*C.float)(unsafe.Pointer(&out32[0])),
		C.int(n),
	)

	// Convert back to float64
	out := NewTensor(m, n)
	for i := range out32 {
		out.data[i] = float64(out32[i])
	}

	return out, nil
}
