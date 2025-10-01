//go:build linux && arm64 && cgo

package main

/*
#cgo CXXFLAGS: -I/usr/include/arm_compute -O3 -std=c++14
#cgo LDFLAGS: -larm_compute -larm_compute_core -lstdc++

#include <stdlib.h>
#include <string.h>

// ARM Compute Library C wrapper
// We need C wrappers because ACL is C++

typedef void* ACLContext;
typedef void* ACLTensor;
typedef void* ACLGEMM;

// Initialize ARM Compute Library context
ACLContext acl_init(void);

// Create tensor descriptor
ACLTensor acl_create_tensor(ACLContext ctx, int rows, int cols);

// Set tensor data
void acl_set_tensor_data(ACLTensor tensor, double* data, int size);

// Get tensor data
void acl_get_tensor_data(ACLTensor tensor, double* data, int size);

// Create GEMM operation
ACLGEMM acl_create_gemm(ACLContext ctx, ACLTensor a, ACLTensor b, ACLTensor c);

// Run GEMM
void acl_run_gemm(ACLGEMM gemm);

// Cleanup
void acl_destroy_tensor(ACLTensor tensor);
void acl_destroy_gemm(ACLGEMM gemm);
void acl_destroy_context(ACLContext ctx);

// Check if ACL is available
int acl_available(void);

// Get ACL info
const char* acl_get_info(void);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ARMCLBackend implements matrix operations using ARM Compute Library
type ARMCLBackend struct {
	available bool
	ctx       C.ACLContext
}

// NewARMCLBackend creates a new ARM Compute Library backend
func NewARMCLBackend() (*ARMCLBackend, error) {
	// Check if ACL is available
	if C.acl_available() == 0 {
		return &ARMCLBackend{available: false},
			fmt.Errorf("ARM Compute Library not available (install: apt-get install libarmcl-dev)")
	}

	ctx := C.acl_init()
	if ctx == nil {
		return &ARMCLBackend{available: false},
			fmt.Errorf("failed to initialize ARM Compute Library")
	}

	return &ARMCLBackend{
		available: true,
		ctx:       ctx,
	}, nil
}

// IsAvailable returns whether ARM Compute Library is available
func (a *ARMCLBackend) IsAvailable() bool {
	return a.available
}

// DeviceName returns the device name
func (a *ARMCLBackend) DeviceName() string {
	if !a.available {
		return "ARM Compute Library (not available)"
	}
	return "ARM Compute Library (optimized for ARM Neoverse/Cortex)"
}

// MatMul performs matrix multiplication using ARM Compute Library
func (a *ARMCLBackend) MatMul(m1, m2 *Tensor) (*Tensor, error) {
	if !a.available {
		return nil, fmt.Errorf("ARM Compute Library not available")
	}

	// Validate dimensions
	if len(m1.shape) != 2 || len(m2.shape) != 2 {
		return nil, fmt.Errorf("ARM CL MatMul requires 2D tensors")
	}

	m, k1 := m1.shape[0], m1.shape[1]
	k2, n := m2.shape[0], m2.shape[1]

	if k1 != k2 {
		return nil, fmt.Errorf("ARM CL MatMul: incompatible dimensions: A is %dx%d, B is %dx%d",
			m, k1, k2, n)
	}

	// Create tensors
	tensorA := C.acl_create_tensor(a.ctx, C.int(m), C.int(k1))
	tensorB := C.acl_create_tensor(a.ctx, C.int(k1), C.int(n))
	tensorC := C.acl_create_tensor(a.ctx, C.int(m), C.int(n))

	if tensorA == nil || tensorB == nil || tensorC == nil {
		return nil, fmt.Errorf("failed to create ACL tensors")
	}

	defer C.acl_destroy_tensor(tensorA)
	defer C.acl_destroy_tensor(tensorB)
	defer C.acl_destroy_tensor(tensorC)

	// Set input data
	C.acl_set_tensor_data(tensorA, (*C.double)(unsafe.Pointer(&m1.data[0])), C.int(len(m1.data)))
	C.acl_set_tensor_data(tensorB, (*C.double)(unsafe.Pointer(&m2.data[0])), C.int(len(m2.data)))

	// Create and run GEMM
	gemm := C.acl_create_gemm(a.ctx, tensorA, tensorB, tensorC)
	if gemm == nil {
		return nil, fmt.Errorf("failed to create ACL GEMM operation")
	}
	defer C.acl_destroy_gemm(gemm)

	C.acl_run_gemm(gemm)

	// Get result
	result := NewTensor(m, n)
	C.acl_get_tensor_data(tensorC, (*C.double)(unsafe.Pointer(&result.data[0])), C.int(len(result.data)))

	return result, nil
}

// GetInfo returns ARM Compute Library information
func (a *ARMCLBackend) GetInfo() string {
	if !a.available {
		return "ARM Compute Library not available"
	}

	info := C.acl_get_info()
	return C.GoString(info)
}

// Close cleans up ARM Compute Library backend resources
func (a *ARMCLBackend) Close() error {
	if a.ctx != nil {
		C.acl_destroy_context(a.ctx)
		a.ctx = nil
	}
	return nil
}
