//go:build linux && cgo

package main

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcublas -lm
#cgo CFLAGS: -I/usr/local/cuda/include

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Check if CUDA is available
int cuda_available(void) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0) ? 1 : 0;
}

// Get CUDA device count
int cuda_device_count(void) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

// Get device properties
typedef struct {
    char name[256];
    size_t totalGlobalMem;
    int major;
    int minor;
    int multiProcessorCount;
    int clockRate;
    size_t sharedMemPerBlock;
} CUDADeviceProps;

int cuda_get_device_props(int device, CUDADeviceProps* props) {
    struct cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, device);
    if (error != cudaSuccess) {
        return 0;
    }

    strncpy(props->name, deviceProp.name, 255);
    props->name[255] = '\0';
    props->totalGlobalMem = deviceProp.totalGlobalMem;
    props->major = deviceProp.major;
    props->minor = deviceProp.minor;
    props->multiProcessorCount = deviceProp.multiProcessorCount;
    props->clockRate = deviceProp.clockRate;
    props->sharedMemPerBlock = deviceProp.sharedMemPerBlock;

    return 1;
}

// cuBLAS handle wrapper
typedef struct {
    cublasHandle_t handle;
    void* d_a;
    void* d_b;
    void* d_c;
    size_t allocated_size;
} CUBLASContext;

// Initialize cuBLAS
CUBLASContext* cublas_init(void) {
    CUBLASContext* ctx = (CUBLASContext*)malloc(sizeof(CUBLASContext));
    if (ctx == NULL) return NULL;

    ctx->d_a = NULL;
    ctx->d_b = NULL;
    ctx->d_c = NULL;
    ctx->allocated_size = 0;

    cublasStatus_t status = cublasCreate(&ctx->handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        free(ctx);
        return NULL;
    }

    return ctx;
}

// Cleanup cuBLAS
void cublas_cleanup(CUBLASContext* ctx) {
    if (ctx == NULL) return;

    if (ctx->d_a) cudaFree(ctx->d_a);
    if (ctx->d_b) cudaFree(ctx->d_b);
    if (ctx->d_c) cudaFree(ctx->d_c);

    cublasDestroy(ctx->handle);
    free(ctx);
}

// Matrix multiplication using cuBLAS DGEMM
// C = alpha * A * B + beta * C
// A is m×k, B is k×n, C is m×n
int cublas_dgemm(CUBLASContext* ctx,
                 double* h_a, double* h_b, double* h_c,
                 int m, int n, int k) {
    if (ctx == NULL) return 0;

    size_t size_a = m * k * sizeof(double);
    size_t size_b = k * n * sizeof(double);
    size_t size_c = m * n * sizeof(double);
    size_t total_size = size_a + size_b + size_c;

    // Allocate or reallocate GPU memory if needed
    if (total_size > ctx->allocated_size) {
        if (ctx->d_a) cudaFree(ctx->d_a);
        if (ctx->d_b) cudaFree(ctx->d_b);
        if (ctx->d_c) cudaFree(ctx->d_c);

        if (cudaMalloc(&ctx->d_a, size_a) != cudaSuccess) return 0;
        if (cudaMalloc(&ctx->d_b, size_b) != cudaSuccess) return 0;
        if (cudaMalloc(&ctx->d_c, size_c) != cudaSuccess) return 0;

        ctx->allocated_size = total_size;
    }

    // Copy data to GPU
    if (cudaMemcpy(ctx->d_a, h_a, size_a, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(ctx->d_b, h_b, size_b, cudaMemcpyHostToDevice) != cudaSuccess) return 0;

    // Perform matrix multiplication
    // cuBLAS uses column-major order, but we're in row-major
    // So we compute: C^T = B^T * A^T
    double alpha = 1.0;
    double beta = 0.0;

    cublasStatus_t status = cublasDgemm(
        ctx->handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,           // Note: swapped dimensions for row-major
        &alpha,
        (double*)ctx->d_b, n,
        (double*)ctx->d_a, k,
        &beta,
        (double*)ctx->d_c, n
    );

    if (status != CUBLAS_STATUS_SUCCESS) return 0;

    // Copy result back to host
    if (cudaMemcpy(h_c, ctx->d_c, size_c, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;

    // Synchronize to ensure completion
    cudaDeviceSynchronize();

    return 1;
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// CUDABackend implements matrix operations using NVIDIA CUDA
type CUDABackend struct {
	available  bool
	deviceID   int
	deviceName string
	ctx        *C.CUBLASContext
	props      DeviceProperties
}

// DeviceProperties holds GPU device properties
type DeviceProperties struct {
	Name                string
	TotalGlobalMem      uint64
	ComputeCapability   string
	MultiProcessorCount int
	ClockRate           int
	SharedMemPerBlock   uint64
}

// NewCUDABackend creates a new CUDA backend
func NewCUDABackend() (*CUDABackend, error) {
	// Check if CUDA is available
	if C.cuda_available() == 0 {
		return &CUDABackend{available: false},
			fmt.Errorf("CUDA not available (no NVIDIA GPU or CUDA runtime not installed)")
	}

	deviceCount := int(C.cuda_device_count())
	if deviceCount == 0 {
		return &CUDABackend{available: false},
			fmt.Errorf("no CUDA devices found")
	}

	// Get properties for device 0
	var cProps C.CUDADeviceProps
	if C.cuda_get_device_props(0, &cProps) == 0 {
		return &CUDABackend{available: false},
			fmt.Errorf("failed to get CUDA device properties")
	}

	props := DeviceProperties{
		Name:                C.GoString(&cProps.name[0]),
		TotalGlobalMem:      uint64(cProps.totalGlobalMem),
		ComputeCapability:   fmt.Sprintf("%d.%d", cProps.major, cProps.minor),
		MultiProcessorCount: int(cProps.multiProcessorCount),
		ClockRate:           int(cProps.clockRate),
		SharedMemPerBlock:   uint64(cProps.sharedMemPerBlock),
	}

	// Initialize cuBLAS
	ctx := C.cublas_init()
	if ctx == nil {
		return &CUDABackend{available: false},
			fmt.Errorf("failed to initialize cuBLAS")
	}

	return &CUDABackend{
		available:  true,
		deviceID:   0,
		deviceName: props.Name,
		ctx:        ctx,
		props:      props,
	}, nil
}

// IsAvailable returns whether CUDA is available
func (c *CUDABackend) IsAvailable() bool {
	return c.available
}

// DeviceName returns the GPU device name
func (c *CUDABackend) DeviceName() string {
	if !c.available {
		return "CUDA (not available)"
	}
	return fmt.Sprintf("%s (CUDA %s, %d SMs)",
		c.props.Name, c.props.ComputeCapability, c.props.MultiProcessorCount)
}

// MatMul performs matrix multiplication using CUDA
func (c *CUDABackend) MatMul(a, b *Tensor) (*Tensor, error) {
	if !c.available {
		return nil, fmt.Errorf("CUDA not available")
	}

	// Validate dimensions
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("CUDA MatMul requires 2D tensors")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]

	if k1 != k2 {
		return nil, fmt.Errorf("CUDA MatMul: incompatible dimensions: A is %dx%d, B is %dx%d",
			m, k1, k2, n)
	}

	k := k1

	// Create result tensor
	result := NewTensor(m, n)

	// Call cuBLAS DGEMM
	success := C.cublas_dgemm(
		c.ctx,
		(*C.double)(unsafe.Pointer(&a.data[0])),
		(*C.double)(unsafe.Pointer(&b.data[0])),
		(*C.double)(unsafe.Pointer(&result.data[0])),
		C.int(m),
		C.int(n),
		C.int(k),
	)

	if success == 0 {
		return nil, fmt.Errorf("cuBLAS DGEMM failed")
	}

	return result, nil
}

// GetInfo returns CUDA device information
func (c *CUDABackend) GetInfo() string {
	if !c.available {
		return "CUDA not available"
	}

	return fmt.Sprintf("%s\n"+
		"  Memory: %.2f GB\n"+
		"  Compute Capability: %s\n"+
		"  SM Count: %d\n"+
		"  Clock Rate: %.2f GHz\n"+
		"  Shared Memory/Block: %d KB",
		c.props.Name,
		float64(c.props.TotalGlobalMem)/(1024*1024*1024),
		c.props.ComputeCapability,
		c.props.MultiProcessorCount,
		float64(c.props.ClockRate)/1e6,
		c.props.SharedMemPerBlock/1024,
	)
}

// Close cleans up CUDA backend resources
func (c *CUDABackend) Close() error {
	if c.ctx != nil {
		C.cublas_cleanup(c.ctx)
		c.ctx = nil
	}
	return nil
}
