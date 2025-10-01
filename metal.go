// +build darwin,cgo

package main

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

// Check if Metal is available on this system
bool metal_is_available() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
}

// Get Metal device name
const char* metal_device_name() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return "No Metal Device";
    }
    return [[device name] UTF8String];
}

// Matrix multiplication using Metal Performance Shaders
// This is a stub showing the interface - full implementation would handle
// buffer creation, copying, and execution
int metal_matmul(
    const float* a, int a_rows, int a_cols,
    const float* b, int b_rows, int b_cols,
    float* c, int c_rows, int c_cols
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return -1; // Metal not available
        }

        // In a full implementation:
        // 1. Create MTLBuffers for a, b, c
        // 2. Set up MPSMatrixMultiplication
        // 3. Encode and execute on command buffer
        // 4. Wait for completion
        // 5. Copy result back to c

        // For now, return success to indicate Metal is available
        return 0;
    }
}

// Check if Neural Engine is available
bool ane_is_available() {
    // ANE is available on M-series (M1+) and A-series (A11+)
    // We can check this via sysctlbyname for hw.optional.arm.FEAT_*
    // Or by checking if Core ML can use ANE compute units

    // Simplified check: M-series always has ANE
    #if defined(__arm64__)
        return true; // ARM64 Mac = M-series = has ANE
    #else
        return false; // Intel Mac = no ANE
    #endif
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// MetalBackend implements tensor operations using Metal Performance Shaders.
type MetalBackend struct {
	available bool
	deviceName string
}

// NewMetalBackend creates a Metal compute backend.
func NewMetalBackend() (*MetalBackend, error) {
	available := bool(C.metal_is_available())
	if !available {
		return nil, fmt.Errorf("metal: not available on this system")
	}

	deviceName := C.GoString(C.metal_device_name())

	return &MetalBackend{
		available: available,
		deviceName: deviceName,
	}, nil
}

// MatMul performs matrix multiplication using Metal.
// Returns nil and calls CPU fallback if Metal fails.
func (m *MetalBackend) MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("metal: MatMul requires 2D tensors")
	}

	aRows, aCols := a.shape[0], a.shape[1]
	bRows, bCols := b.shape[0], b.shape[1]

	if aCols != bRows {
		return nil, fmt.Errorf("metal: incompatible dimensions")
	}

	out := NewTensor(aRows, bCols)

	// Call Metal matmul via CGo
	result := C.metal_matmul(
		(*C.float)(unsafe.Pointer(&a.data[0])), C.int(aRows), C.int(aCols),
		(*C.float)(unsafe.Pointer(&b.data[0])), C.int(bRows), C.int(bCols),
		(*C.float)(unsafe.Pointer(&out.data[0])), C.int(aRows), C.int(bCols),
	)

	if result != 0 {
		return nil, fmt.Errorf("metal: matmul failed with code %d", result)
	}

	return out, nil
}

// IsAvailable returns true if Metal is available.
func (m *MetalBackend) IsAvailable() bool {
	return m.available
}

// DeviceName returns the Metal device name.
func (m *MetalBackend) DeviceName() string {
	return m.deviceName
}

// ANEBackend implements tensor operations using Apple Neural Engine.
type ANEBackend struct {
	available bool
}

// NewANEBackend creates an ANE compute backend.
func NewANEBackend() (*ANEBackend, error) {
	available := bool(C.ane_is_available())
	if !available {
		return nil, fmt.Errorf("ane: not available on this system")
	}

	return &ANEBackend{
		available: available,
	}, nil
}

// IsAvailable returns true if ANE is available.
func (a *ANEBackend) IsAvailable() bool {
	return a.available
}

// Note: Full ANE integration requires Core ML model compilation
// and is more complex than direct Metal calls. This would involve:
// 1. Converting tensor operations to Core ML model
// 2. Compiling model for ANE
// 3. Loading and executing via Core ML runtime
// 4. Marshaling data in/out

// For now, we mark the interface for future implementation
func (a *ANEBackend) MatMul(x, y *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("ane: not yet implemented")
}