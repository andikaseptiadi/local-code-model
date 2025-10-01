//go:build linux && arm64 && cgo

package main

/*
#cgo CFLAGS: -march=armv8-a+simd -O3
#cgo LDFLAGS: -lm

#include <stdint.h>

// Forward declarations from matmul_neon.c
void matmul_neon_c(double* c, const double* a, const double* b, int64_t m, int64_t n, int64_t k);
void matmul_neon_optimized(double* c, const double* a, const double* b, int64_t m, int64_t n, int64_t k);
int neon_available(void);
*/
import "C"
import (
	"unsafe"
)

// MatMulSIMD performs matrix multiplication using NEON SIMD vectorization.
// NEON is mandatory on ARM64, so this always works on Linux ARM64.
func MatMulSIMD(a, b *Tensor) *Tensor {
	shapeA := a.Shape()
	shapeB := b.Shape()
	m, k := shapeA[0], shapeA[1]
	n := shapeB[1]

	result := NewTensor(m, n)

	// Call optimized NEON implementation
	C.matmul_neon_optimized(
		(*C.double)(unsafe.Pointer(&result.data[0])),
		(*C.double)(unsafe.Pointer(&a.data[0])),
		(*C.double)(unsafe.Pointer(&b.data[0])),
		C.int64_t(m),
		C.int64_t(n),
		C.int64_t(k),
	)

	return result
}

// IsSIMDAvailable returns true on ARM64 (NEON is mandatory)
func IsSIMDAvailable() bool {
	return C.neon_available() == 1
}

// SIMDInfo returns information about SIMD availability.
func SIMDInfo() string {
	return "ARM64 NEON (C intrinsics, 128-bit vectors)"
}
