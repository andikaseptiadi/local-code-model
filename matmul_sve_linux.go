//go:build linux && arm64

package main

/*
// Try to enable SVE if supported, but don't fail if not
// The C code has fallback for non-SVE
#cgo CFLAGS: -O3 -march=native -fopenmp
#cgo LDFLAGS: -lm -lgomp

#include <stdint.h>

// Forward declarations from matmul_sve.c
void matmul_sve_c(double* c, const double* a, const double* b, int64_t m, int64_t n, int64_t k);
void matmul_sve_optimized(double* c, const double* a, const double* b, int64_t m, int64_t n, int64_t k);
int64_t sve_vector_length(void);

// Forward declarations from matmul_sve_multiengine.c
void matmul_sve_multiengine(double* c, const double* a, const double* b, int64_t m, int64_t n, int64_t k);
void matmul_sve_multiengine_pinned(double* c, const double* a, const double* b, int64_t m, int64_t n, int64_t k);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// SVEBackend implements matrix operations using ARM SVE
type SVEBackend struct {
	available    bool
	vectorLength int
	hasSVE2      bool
}

// NewSVEBackend creates a new SVE backend
func NewSVEBackend() (*SVEBackend, error) {
	features := DetectCPUFeatures()

	if !features.HasSVE {
		return &SVEBackend{available: false}, fmt.Errorf("SVE not available on this CPU (requires Graviton3+)")
	}

	vl := int(C.sve_vector_length())
	if vl == 0 {
		return &SVEBackend{available: false}, fmt.Errorf("SVE not available (compiler support missing)")
	}

	return &SVEBackend{
		available:    true,
		vectorLength: vl,
		hasSVE2:      features.HasSVE2,
	}, nil
}

// IsAvailable returns whether SVE is available
func (s *SVEBackend) IsAvailable() bool {
	return s.available
}

// DeviceName returns the device name
func (s *SVEBackend) DeviceName() string {
	if !s.available {
		return "SVE (not available)"
	}
	sveVersion := "SVE"
	if s.hasSVE2 {
		sveVersion = "SVE2"
	}
	return fmt.Sprintf("ARM %s (%d-bit vectors, %d float64 elements)",
		sveVersion, s.vectorLength*64, s.vectorLength)
}

// MatMul performs matrix multiplication using SVE
func (s *SVEBackend) MatMul(a, b *Tensor) (*Tensor, error) {
	if !s.available {
		return nil, fmt.Errorf("SVE not available")
	}

	// Validate dimensions
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("SVE MatMul requires 2D tensors")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]

	if k1 != k2 {
		return nil, fmt.Errorf("SVE MatMul: incompatible dimensions: A is %dx%d, B is %dx%d",
			m, k1, k2, n)
	}

	k := k1

	// Create result tensor
	result := NewTensor(m, n)

	// Call SVE C implementation
	C.matmul_sve_optimized(
		(*C.double)(unsafe.Pointer(&result.data[0])),
		(*C.double)(unsafe.Pointer(&a.data[0])),
		(*C.double)(unsafe.Pointer(&b.data[0])),
		C.int64_t(m),
		C.int64_t(n),
		C.int64_t(k),
	)

	return result, nil
}

// VectorLength returns the SVE vector length in float64 elements
func (s *SVEBackend) VectorLength() int {
	return s.vectorLength
}

// HasSVE2 returns whether SVE2 is available
func (s *SVEBackend) HasSVE2() bool {
	return s.hasSVE2
}

// MatMulMultiEngine performs matrix multiplication using multi-engine optimized SVE
func (s *SVEBackend) MatMulMultiEngine(a, b *Tensor) (*Tensor, error) {
	if !s.available {
		return nil, fmt.Errorf("SVE not available")
	}

	// Validate dimensions
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("SVE MatMul requires 2D tensors")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]

	if k1 != k2 {
		return nil, fmt.Errorf("SVE MatMul: incompatible dimensions: A is %dx%d, B is %dx%d",
			m, k1, k2, n)
	}

	k := k1

	// Create result tensor
	result := NewTensor(m, n)

	// Call multi-engine optimized implementation
	C.matmul_sve_multiengine(
		(*C.double)(unsafe.Pointer(&result.data[0])),
		(*C.double)(unsafe.Pointer(&a.data[0])),
		(*C.double)(unsafe.Pointer(&b.data[0])),
		C.int64_t(m),
		C.int64_t(n),
		C.int64_t(k),
	)

	return result, nil
}

// MatMulMultiEnginePinned performs matrix multiplication using pinned multi-engine SVE
func (s *SVEBackend) MatMulMultiEnginePinned(a, b *Tensor) (*Tensor, error) {
	if !s.available {
		return nil, fmt.Errorf("SVE not available")
	}

	// Validate dimensions
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("SVE MatMul requires 2D tensors")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]

	if k1 != k2 {
		return nil, fmt.Errorf("SVE MatMul: incompatible dimensions: A is %dx%d, B is %dx%d",
			m, k1, k2, n)
	}

	k := k1

	// Create result tensor
	result := NewTensor(m, n)

	// Call pinned multi-engine implementation
	C.matmul_sve_multiengine_pinned(
		(*C.double)(unsafe.Pointer(&result.data[0])),
		(*C.double)(unsafe.Pointer(&a.data[0])),
		(*C.double)(unsafe.Pointer(&b.data[0])),
		C.int64_t(m),
		C.int64_t(n),
		C.int64_t(k),
	)

	return result, nil
}

// Close cleans up SVE backend resources
func (s *SVEBackend) Close() error {
	return nil // Nothing to clean up
}
