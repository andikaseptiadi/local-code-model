//go:build linux && cgo

package main

/*
#cgo LDFLAGS: -lopenblas -lm
#include <stdint.h>

// CBLAS interface for DGEMM
// C = alpha * A * B + beta * C
// where A is m×k, B is k×n, C is m×n
void cblas_dgemm(
    int Order,      // 101=RowMajor, 102=ColMajor
    int TransA,     // 111=NoTrans, 112=Trans
    int TransB,
    int M,          // rows of A and C
    int N,          // cols of B and C
    int K,          // cols of A, rows of B
    double alpha,
    const double *A,
    int lda,        // leading dimension of A
    const double *B,
    int ldb,
    double beta,
    double *C,
    int ldc
);

// Check if OpenBLAS is available
int openblas_available(void) {
    return 1;  // If this compiles and links, OpenBLAS is available
}

// Get OpenBLAS version
void openblas_get_config(void);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// OpenBLASBackend implements matrix operations using OpenBLAS
type OpenBLASBackend struct {
	available bool
}

// NewOpenBLASBackend creates a new OpenBLAS backend
func NewOpenBLASBackend() (*OpenBLASBackend, error) {
	// Check if OpenBLAS is available
	avail := C.openblas_available()
	if avail == 0 {
		return &OpenBLASBackend{available: false},
			fmt.Errorf("OpenBLAS not available (install: apt-get install libopenblas-dev)")
	}

	return &OpenBLASBackend{available: true}, nil
}

// IsAvailable returns whether OpenBLAS is available
func (o *OpenBLASBackend) IsAvailable() bool {
	return o.available
}

// DeviceName returns the device name
func (o *OpenBLASBackend) DeviceName() string {
	if !o.available {
		return "OpenBLAS (not available)"
	}
	return "OpenBLAS (optimized BLAS for ARM64/x86_64)"
}

// MatMul performs matrix multiplication using OpenBLAS
// This is typically 10-20× faster than naive implementations
func (o *OpenBLASBackend) MatMul(a, b *Tensor) (*Tensor, error) {
	if !o.available {
		return nil, fmt.Errorf("OpenBLAS not available")
	}

	// Validate dimensions
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("OpenBLAS MatMul requires 2D tensors")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]

	if k1 != k2 {
		return nil, fmt.Errorf("OpenBLAS MatMul: incompatible dimensions: A is %dx%d, B is %dx%d",
			m, k1, k2, n)
	}

	k := k1

	// Create result tensor
	result := NewTensor(m, n)

	// Call CBLAS DGEMM
	// C = alpha * A * B + beta * C
	// We want C = A * B, so alpha=1.0, beta=0.0
	const (
		CblasRowMajor = 101
		CblasNoTrans  = 111
	)

	C.cblas_dgemm(
		CblasRowMajor, // Row-major order
		CblasNoTrans,  // Don't transpose A
		CblasNoTrans,  // Don't transpose B
		C.int(m),      // rows of A
		C.int(n),      // cols of B
		C.int(k),      // cols of A / rows of B
		1.0,           // alpha
		(*C.double)(unsafe.Pointer(&a.data[0])),
		C.int(k),      // leading dimension of A
		(*C.double)(unsafe.Pointer(&b.data[0])),
		C.int(n),      // leading dimension of B
		0.0,           // beta
		(*C.double)(unsafe.Pointer(&result.data[0])),
		C.int(n),      // leading dimension of C
	)

	return result, nil
}

// Close cleans up OpenBLAS backend resources
func (o *OpenBLASBackend) Close() error {
	return nil // Nothing to clean up
}

// GetInfo returns OpenBLAS configuration information
func (o *OpenBLASBackend) GetInfo() string {
	if !o.available {
		return "OpenBLAS not available"
	}

	// Call openblas_get_config to print version info
	// This prints to stdout, so we just return a message
	return "OpenBLAS available - use OPENBLAS_VERBOSE=1 for details"
}
