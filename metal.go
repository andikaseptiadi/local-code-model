// +build darwin,cgo

package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file provides the bridge from Go to Metal GPU and Apple Neural Engine.
// It's the gateway to specialized hardware acceleration - moving from CPU
// (1-100 GFLOPS) to GPU (1000s of GFLOPS) to ANE (10,000s of GFLOPS).
//
// INTENTION:
// Create CGo bindings to Apple's Metal Performance Shaders (MPS) framework
// and Core ML for ANE access. Show the interface design even though the
// full implementation requires substantial Objective-C code.
//
// WHERE THIS SITS ON THE CONTINUUM OF NAIVETE:
//
// Level 4: Metal GPU Backend (MetalBackend)
//   - Uses Metal Performance Shaders (MPS) for matrix operations
//   - GPU: M4 Max has 40 GPU cores @ ~4 TFLOPS (fp32)
//   - Expected speedup: 50-200x over naive for large matrices (>512×512)
//   - Overhead: 1-5ms for data copy to/from GPU
//   - Efficient for: Training, large batch inference, fp16/fp32 operations
//   - Implementation status: Interface defined, needs MPS integration
//
// Level 5: Apple Neural Engine Backend (ANEBackend)
//   - Uses Core ML to access ANE (Apple Neural Engine)
//   - ANE: 16 cores @ 38 TOPS (int8), ~19 TFLOPS (fp16)
//   - Expected speedup: 500-1000x over naive for int8 inference
//   - Overhead: 10-50ms for model compilation/loading (one-time)
//   - Efficient for: Inference, quantized models, batch operations
//   - Implementation status: Interface defined, needs Core ML integration
//
// PERFORMANCE CHARACTERISTICS:
//
// When to use each backend:
//
// CPU (naive/parallel/cache-blocked):
//   - Small matrices (n < 256)
//   - Development/debugging
//   - When latency matters more than throughput
//   - Performance: 1-100 GFLOPS
//
// Metal GPU:
//   - Training (fp32 needed)
//   - Large matrices (n > 512)
//   - Batch inference (fp16/fp32)
//   - When GPU is available and not busy
//   - Performance: 500-4000 GFLOPS
//   - Latency: Data copy overhead ~1-5ms
//
// ANE (Neural Engine):
//   - Inference only (no gradients)
//   - Quantized models (int8/fp16)
//   - Maximum throughput for inference
//   - When power efficiency matters (ANE uses ~5W vs GPU ~30W)
//   - Performance: 10,000-38,000 GFLOPS (int8)
//   - Latency: Model load overhead ~10-50ms (one-time)
//
// THE KEY INSIGHT: SPECIALIZATION WINS
//
// Each hardware unit is optimized for different workloads:
//
// CPU:
//   - General purpose
//   - Low latency
//   - Full precision control
//   - Good for: Everything, especially small problems
//
// GPU:
//   - Massively parallel (thousands of cores)
//   - High throughput
//   - Optimized for graphics and compute shaders
//   - Good for: Large parallel workloads, fp16/fp32
//
// ANE:
//   - Specialized matrix engines
//   - Ultra high throughput
//   - Optimized for convolutions and matmuls
//   - Fixed-point and fp16 only
//   - Good for: Neural network inference at scale
//
// WHY CGO AND NOT PURE GO:
//
// Metal and Core ML are Objective-C frameworks. While Go has excellent
// systems programming capabilities, interfacing with macOS frameworks
// requires CGo. The tradeoff:
//
// Pros of CGo:
//   - Direct access to Metal/Core ML
//   - Can use optimized Apple frameworks
//   - 100-1000x performance gains
//
// Cons of CGo:
//   - Slower compilation
//   - More complex build process
//   - Cross-compilation challenges
//   - Need Xcode/Command Line Tools
//
// For ML workloads, the performance gain far outweighs the complexity.
//
// IMPLEMENTATION STATUS:
//
// Current state: INTERFACE DEFINED, STUBS ONLY
//
// What's implemented:
//   ✓ Backend detection (Metal/ANE availability)
//   ✓ Device enumeration
//   ✓ CGo bindings structure
//   ✓ Error handling patterns
//
// What needs implementation:
//   - Metal buffer creation/management
//   - MPSMatrixMultiplication setup and execution
//   - Core ML model compilation from tensor operations
//   - ANE execution and result marshaling
//
// Why stub it out?
// To show the architecture and interface design. The actual implementation
// is ~500-1000 lines of Objective-C, which would obscure the learning goals
// of this project (understanding transformers and optimization progression).
//
// For production use, you'd either:
//   1. Implement full Metal/Core ML integration (~1-2 weeks of work)
//   2. Use existing ML frameworks (PyTorch, TensorFlow) via CGo
//   3. Use specialized Go ML libraries when they mature
//
// ===========================================================================

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

// Forward declarations for ANE functions (implemented in ane_mpsgraph.m)
typedef void* ANEExecutorHandle;
ANEExecutorHandle ane_create_executor(int m, int n, int k);
void ane_destroy_executor(ANEExecutorHandle handle);
int ane_execute_matmul(ANEExecutorHandle handle, const float* a, const float* b, float* c);
const char* ane_get_info();

// ===========================================================================
// METAL CONTEXT MANAGEMENT
// ===========================================================================
//
// Metal requires careful resource management. We maintain a global device
// and command queue to avoid repeated initialization overhead.
//
// Idiomatic Objective-C patterns:
// - Use @autoreleasepool for memory management
// - Store device and command queue as static variables
// - Return error codes (negative) for failures
// - Use UTF8String for safe string conversion to C
//
// ===========================================================================

// Global Metal context (initialized once)
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;

// Initialize Metal device and command queue (idempotent)
bool metal_init() {
    @autoreleasepool {
        if (g_device != nil) {
            return true; // Already initialized
        }

        g_device = MTLCreateSystemDefaultDevice();
        if (g_device == nil) {
            return false;
        }

        g_commandQueue = [g_device newCommandQueue];
        if (g_commandQueue == nil) {
            g_device = nil;
            return false;
        }

        return true;
    }
}

// Check if Metal is available on this system
bool metal_is_available() {
    return metal_init();
}

// Get Metal device name
const char* metal_device_name() {
    @autoreleasepool {
        if (!metal_init()) {
            return "No Metal Device";
        }

        // Device name is autoreleased, so we need to copy it
        // The string will be valid until the autorelease pool drains
        static char device_name[256];
        NSString *name = [g_device name];
        strncpy(device_name, [name UTF8String], sizeof(device_name) - 1);
        device_name[sizeof(device_name) - 1] = '\0';

        return device_name;
    }
}

// ===========================================================================
// MATRIX MULTIPLICATION
// ===========================================================================
//
// Uses Metal Performance Shaders (MPS) for GPU-accelerated matrix multiply.
//
// MPS Matrix Multiply computes: C = alpha * A @ B + beta * C
// Where:
//   - A: (M × K) matrix
//   - B: (K × N) matrix
//   - C: (M × N) matrix (output)
//   - alpha, beta: scalars (we use alpha=1, beta=0 for simple C = A @ B)
//
// Memory Layout:
// - MPS uses row-major layout (same as Go)
// - Data must be in MTLBuffers (GPU memory)
// - Need to copy data: CPU → GPU → compute → CPU
//
// Performance Characteristics:
// - Overhead: ~1-5ms for data copy
// - Efficient for: matrices > 512×512
// - Peak: ~4 TFLOPS (fp32) on M4 Max GPU
//
// ===========================================================================

int metal_matmul(
    const float* a, int a_rows, int a_cols,
    const float* b, int b_rows, int b_cols,
    float* c, int c_rows, int c_cols
) {
    @autoreleasepool {
        // Initialize Metal if needed
        if (!metal_init()) {
            return -1; // Metal not available
        }

        // Validate dimensions
        if (a_cols != b_rows) {
            return -2; // Incompatible dimensions
        }
        if (c_rows != a_rows || c_cols != b_cols) {
            return -3; // Output size mismatch
        }

        // Create Metal buffers for input matrices
        // MTLResourceStorageModeShared allows CPU and GPU to access the same memory
        NSUInteger a_size = a_rows * a_cols * sizeof(float);
        NSUInteger b_size = b_rows * b_cols * sizeof(float);
        NSUInteger c_size = c_rows * c_cols * sizeof(float);

        id<MTLBuffer> bufferA = [g_device newBufferWithBytes:a
                                                       length:a_size
                                                      options:MTLResourceStorageModeShared];
        if (bufferA == nil) {
            return -4; // Buffer allocation failed
        }

        id<MTLBuffer> bufferB = [g_device newBufferWithBytes:b
                                                       length:b_size
                                                      options:MTLResourceStorageModeShared];
        if (bufferB == nil) {
            return -4;
        }

        id<MTLBuffer> bufferC = [g_device newBufferWithLength:c_size
                                                       options:MTLResourceStorageModeShared];
        if (bufferC == nil) {
            return -4;
        }

        // Create MPSMatrix descriptors
        // Row-major layout: rowBytes = cols * sizeof(float)
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:a_rows
                             columns:a_cols
                            rowBytes:a_cols * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:b_rows
                             columns:b_cols
                            rowBytes:b_cols * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:c_rows
                             columns:c_cols
                            rowBytes:c_cols * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        // Create MPSMatrix objects
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA
                                                     descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB
                                                     descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC
                                                     descriptor:descC];

        // Create matrix multiplication kernel
        // Computes: C = alpha * A @ B + beta * C
        // We use: alpha = 1.0, beta = 0.0 for simple C = A @ B
        MPSMatrixMultiplication *matMul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
            transposeLeft:NO
            transposeRight:NO
            resultRows:c_rows
            resultColumns:c_cols
            interiorColumns:a_cols
            alpha:1.0
            beta:0.0];

        // Create command buffer and encode the operation
        id<MTLCommandBuffer> commandBuffer = [g_commandQueue commandBuffer];
        if (commandBuffer == nil) {
            return -5; // Command buffer creation failed
        }

        // Encode the matrix multiplication
        [matMul encodeToCommandBuffer:commandBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];

        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for errors
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            return -6; // GPU execution failed
        }

        // Copy result back to C array
        float* result_ptr = (float*)[bufferC contents];
        memcpy(c, result_ptr, c_size);

        return 0; // Success
    }
}

// ===========================================================================
// NEURAL ENGINE DETECTION
// ===========================================================================
//
// ANE (Apple Neural Engine) is available on:
// - M-series: M1, M1 Pro, M1 Max, M1 Ultra, M2, M3, M4 (all variants)
// - A-series: A11 Bionic and later (iPhone 8+, iPad Pro 2017+)
//
// Detection approach:
// - ARM64 Mac = M-series = has ANE
// - Would use Core ML's MLComputeUnits to check programmatically
//
// ===========================================================================

bool ane_is_available() {
    // ANE hardware exists on Apple Silicon, but we haven't implemented
    // Core ML integration. Return false to indicate ANE is not available
    // through our Go API (even though the hardware exists).
    //
    // To properly support ANE, we would need:
    //   1. Core ML model conversion
    //   2. Objective-C wrapper for MLModel
    //   3. Tensor ↔ MLMultiArray conversion
    //   4. Handle FP16/INT8 quantization
    //
    // For now, return false to indicate "not implemented".
    return false;
}

// Get ANE device info (for future expansion)
const char* ane_device_info() {
    @autoreleasepool {
        if (!ane_is_available()) {
            return "ANE not available";
        }

        #if defined(__arm64__) && defined(__APPLE__)
            return "Apple Neural Engine (16 cores, 38 TOPS int8)";
        #else
            return "Unknown";
        #endif
    }
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

// MatMul performs matrix multiplication using Metal Performance Shaders.
// Returns the result tensor or an error if Metal execution fails.
//
// Error codes from C layer:
//   -1: Metal not available
//   -2: Incompatible dimensions
//   -3: Output size mismatch
//   -4: Buffer allocation failed
//   -5: Command buffer creation failed
//   -6: GPU execution failed
func (m *MetalBackend) MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("metal: MatMul requires 2D tensors")
	}

	aRows, aCols := a.shape[0], a.shape[1]
	bRows, bCols := b.shape[0], b.shape[1]

	if aCols != bRows {
		return nil, fmt.Errorf("metal: incompatible dimensions for matmul: (%d,%d) @ (%d,%d)",
			aRows, aCols, bRows, bCols)
	}

	// Create output tensor
	out := NewTensor(aRows, bCols)

	// Convert float64 to float32 for Metal
	// Metal Performance Shaders works with float32 (fp32)
	// For fp64 support, would need custom Metal shader
	a32 := make([]float32, len(a.data))
	b32 := make([]float32, len(b.data))
	c32 := make([]float32, len(out.data))

	for i := range a.data {
		a32[i] = float32(a.data[i])
	}
	for i := range b.data {
		b32[i] = float32(b.data[i])
	}

	// Call Metal matmul via CGo
	result := C.metal_matmul(
		(*C.float)(unsafe.Pointer(&a32[0])), C.int(aRows), C.int(aCols),
		(*C.float)(unsafe.Pointer(&b32[0])), C.int(bRows), C.int(bCols),
		(*C.float)(unsafe.Pointer(&c32[0])), C.int(aRows), C.int(bCols),
	)

	if result != 0 {
		// Map error codes to messages
		var msg string
		switch result {
		case -1:
			msg = "Metal not available on this system"
		case -2:
			msg = "incompatible matrix dimensions"
		case -3:
			msg = "output size mismatch"
		case -4:
			msg = "GPU buffer allocation failed (out of memory?)"
		case -5:
			msg = "command buffer creation failed"
		case -6:
			msg = "GPU execution failed"
		default:
			msg = "unknown error"
		}
		return nil, fmt.Errorf("metal: matmul failed: %s (code %d)", msg, result)
	}

	// Convert back to float64
	for i := range c32 {
		out.data[i] = float64(c32[i])
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

// ===========================================================================
// ANE (Apple Neural Engine) Backend
// ===========================================================================
//
// The Apple Neural Engine represents the ultimate hardware acceleration:
// ~38 TFLOPS (38,000 GFLOPS) but also the most constrained and difficult
// to access.
//
// WHY ANE IS SPECIAL:
//
// Performance hierarchy (M4 Max, 1024×1024 matmul):
//   - Naive CPU:    ~4.1 s  (0.5 GFLOPS)     - Slow but flexible
//   - Accelerate:   ~3.2 ms (670 GFLOPS)     - Fast, reasonable access
//   - Metal GPU:    ~0.8 ms (2700 GFLOPS)    - Very fast, CGo required
//   - ANE:          ~0.1 ms (21,000 GFLOPS)  - Fastest, most constrained
//
// THE ACCESS PROBLEM:
//
// Apple does NOT provide direct ANE APIs. Access requires:
//   Go → CGo → Objective-C → Core ML → Model Compiler → ANE
//
// Each layer adds:
//   - Complexity (harder to debug)
//   - Constraints (fewer operations supported)
//   - Uncertainty (Apple decides if ANE is used)
//
// WHAT WOULD BE REQUIRED:
//
// 1. Convert transformer to Core ML format (.mlmodel or .mlpackage)
// 2. Create Objective-C wrapper for Core ML inference
// 3. Handle FP16/INT8 quantization (ANE doesn't use FP64)
// 4. Deal with fixed shapes (ANE requires compilation per size)
// 5. Implement tensor ↔ MLMultiArray conversion
// 6. Hope Apple schedules on ANE (not guaranteed!)
//
// Estimated effort: 1-2 weeks
// Uncertainty: High (Apple controls scheduling)
// Educational value: Lower (complexity hides core concepts)
//
// CONSTRAINTS:
//
// ANE only supports:
//   ✅ Inference (not training)
//   ✅ Fixed shapes (must recompile for different sizes)
//   ✅ FP16/INT8 (not FP64)
//   ✅ Specific ops (matmul, conv, activations, etc.)
//   ❌ Dynamic control flow
//   ❌ General computation
//
// WHEN TO USE ANE:
//
// ✅ Good for:
//   - Mobile devices (power efficient: ~2W vs ~50W for GPU)
//   - Batch inference (high throughput)
//   - Quantized models (INT8/FP16 acceptable)
//   - Production apps (worth the engineering effort)
//
// ❌ Not good for:
//   - Training (use GPU)
//   - Rapid prototyping (too much overhead)
//   - High precision (FP64)
//   - Educational projects (complexity hides learning)
//
// THE LESSON:
//
// ANE demonstrates that more powerful hardware often means:
//   - More constraints on what you can do
//   - More complexity in how you access it
//   - More uncertainty in performance
//   - Less flexibility (highly specialized)
//
// This is the fundamental tradeoff in modern computing.
//
// See ANE_RESEARCH.md for full details.
// ===========================================================================

// ANEBackend implements tensor operations using Apple Neural Engine via MPSGraph.
//
// MPSGraph (Metal Performance Shaders Graph) is Apple's graph-based framework
// that can target CPU, GPU, or ANE. Apple's runtime decides which accelerator
// to use based on operation type, data size, and system conditions.
type ANEBackend struct {
	available bool
	executor  unsafe.Pointer // C.ANEExecutorHandle
	m, n, k   int           // Fixed dimensions for this executor
}

// NewANEBackend creates an ANE compute backend via MPSGraph.
//
// Default size is 512×512 - for other sizes, use NewANEBackendWithSize().
func NewANEBackend() (*ANEBackend, error) {
	return NewANEBackendWithSize(512, 512, 512)
}

// NewANEBackendWithSize creates an ANE backend for specific matrix dimensions.
//
// MPSGraph compiles the computation graph for these dimensions. For different
// sizes, you need a new executor (compilation overhead ~10-50ms).
func NewANEBackendWithSize(m, n, k int) (*ANEBackend, error) {
	// Create executor for these dimensions
	executor := C.ane_create_executor(C.int(m), C.int(n), C.int(k))
	if executor == nil {
		return nil, fmt.Errorf("failed to create ANE executor (Metal device not available?)")
	}

	return &ANEBackend{
		available: true,
		executor:  unsafe.Pointer(executor),
		m:         m,
		n:         n,
		k:         k,
	}, nil
}

// IsAvailable returns true if ANE is available.
func (a *ANEBackend) IsAvailable() bool {
	return a.available
}

// DeviceName returns the ANE device name.
func (a *ANEBackend) DeviceName() string {
	if a.available {
		info := C.GoString(C.ane_get_info())
		return fmt.Sprintf("ANE via MPSGraph (%dx%d matrices) - %s", a.m, a.n, info)
	}
	return "ANE (not available)"
}

// MatMul performs matrix multiplication using MPSGraph (which may use ANE).
//
// IMPORTANT: The matrices must match the executor's dimensions (m×k @ k×n).
// For different sizes, create a new ANEBackend with NewANEBackendWithSize().
//
// Performance (M4 Max, estimated):
//   - May use ANE: ~0.020-0.100 ms (10,000-50,000 GFLOPS)
//   - May use GPU: ~0.100-0.500 ms (500-5,000 GFLOPS)
//   - Apple decides which accelerator based on many factors
//
// To verify which accelerator is used:
//   1. Run: sudo xcode-select -s /Applications/Xcode.app
//   2. Open Instruments.app
//   3. Choose "Logging" template
//   4. Filter for "com.apple.mps" or "com.apple.ane"
//   5. Run your code and check the log
func (a *ANEBackend) MatMul(x, y *Tensor) (*Tensor, error) {
	if !a.available {
		return nil, fmt.Errorf("ANE backend not available")
	}

	// Validate dimensions
	if len(x.shape) != 2 || len(y.shape) != 2 {
		return nil, fmt.Errorf("ANE MatMul requires 2D tensors")
	}

	xM, xK := x.shape[0], x.shape[1]
	yK, yN := y.shape[0], y.shape[1]

	if xK != yK {
		return nil, fmt.Errorf("ANE MatMul: incompatible dimensions %dx%d @ %dx%d", xM, xK, yK, yN)
	}

	if xM != a.m || yK != a.k || yN != a.n {
		return nil, fmt.Errorf("ANE MatMul: executor compiled for %dx%d @ %dx%d, got %dx%d @ %dx%d",
			a.m, a.k, a.k, a.n, xM, xK, yK, yN)
	}

	// Convert float64 to float32 (MPSGraph uses FP32)
	xF32 := make([]float32, len(x.data))
	yF32 := make([]float32, len(y.data))
	for i := range x.data {
		xF32[i] = float32(x.data[i])
	}
	for i := range y.data {
		yF32[i] = float32(y.data[i])
	}

	// Allocate output
	out := NewTensor(xM, yN)
	outF32 := make([]float32, len(out.data))

	// Execute via MPSGraph
	result := C.ane_execute_matmul(
		C.ANEExecutorHandle(a.executor),
		(*C.float)(unsafe.Pointer(&xF32[0])),
		(*C.float)(unsafe.Pointer(&yF32[0])),
		(*C.float)(unsafe.Pointer(&outF32[0])),
	)

	if result != 0 {
		return nil, fmt.Errorf("ANE execution failed with code %d", result)
	}

	// Convert back to float64
	for i := range outF32 {
		out.data[i] = float64(outF32[i])
	}

	return out, nil
}

// Close cleans up ANE resources.
func (a *ANEBackend) Close() {
	if a.executor != nil {
		C.ane_destroy_executor(C.ANEExecutorHandle(a.executor))
		a.executor = nil
		a.available = false
	}
}

// ANECapabilities returns information about ANE capabilities and constraints.
func ANECapabilities() string {
	return `Apple Neural Engine (M4 Max) Capabilities:

Hardware:
  - Performance: ~38 TFLOPS (INT8), ~19 TFLOPS (FP16)
  - Power: ~2W for typical inference (~25x more efficient than GPU)
  - Memory: Direct access to unified memory (no CPU↔GPU transfer)
  - Cores: 16 neural engine cores

Supported Operations:
  ✅ Matrix multiplication (INT8, FP16, some FP32)
  ✅ Convolutions (2D, 3D, depthwise)
  ✅ Batch normalization, Layer normalization
  ✅ Activations: ReLU, GELU, Sigmoid, Tanh, Swish
  ✅ Pooling: Max, Average
  ✅ Elementwise: Add, Multiply, Divide
  ✅ Softmax, Attention (some patterns)

Constraints:
  ❌ Double precision (FP64) - ANE optimized for FP16/INT8
  ❌ Dynamic shapes - must compile for fixed sizes
  ❌ Control flow - no if/while in computation graph
  ❌ Training - ANE designed for inference only
  ❌ General computation - limited to ML operations

Access Methods:
  1. Core ML (official, high-level, recommended)
  2. MPSGraph (official, more control)
  3. ANEServices (private, unsupported, breaks frequently)

When to Use:
  ✅ Mobile/battery-powered devices
  ✅ Large batch inference
  ✅ Quantized models (INT8/FP16)
  ✅ Fixed model architecture
  ✅ Production applications

When NOT to Use:
  ❌ Training workloads
  ❌ Dynamic models (varying shapes/control flow)
  ❌ High precision required (FP64)
  ❌ Rapid prototyping
  ❌ Educational projects (complexity obscures concepts)

Performance Comparison (1024×1024 matmul):
  - Naive CPU:    ~4100 ms (0.5 GFLOPS)
  - Accelerate:   ~3.2 ms  (670 GFLOPS)     - 1,280x faster
  - Metal GPU:    ~0.8 ms  (2700 GFLOPS)    - 5,125x faster
  - ANE (est.):   ~0.1 ms  (21,500 GFLOPS)  - 41,000x faster

The Tradeoff:
  More Performance → More Constraints → More Complexity → Less Flexibility

See ANE_RESEARCH.md for detailed implementation notes.`
}