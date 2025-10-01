// ===========================================================================
// ANE Implementation via MPSGraph
// ===========================================================================
//
// MPSGraph (Metal Performance Shaders Graph) is Apple's graph-based compute
// framework that can target:
//   - CPU
//   - GPU (Metal)
//   - ANE (Apple Neural Engine)
//
// FRAMEWORK STACK (high to low level):
//   1. Core ML (.mlmodel) - Model-based, inference-focused, most ANE control
//   2. MLCompute - DEPRECATED in macOS 13+, was for TensorFlow/PyTorch
//   3. MPSGraph - Current recommended approach (what we use)
//   4. Metal/MPS - GPU-only, no ANE access
//   5. AppleNeuralEngine.framework - Private, not accessible
//
// WHY MPSGraph (not Core ML or MLCompute):
//   - Pure Objective-C API (no Python/model conversion needed)
//   - Can build graphs programmatically (perfect for training)
//   - Current, maintained framework (2024+)
//   - Simpler for basic operations like matmul
//   - MLCompute is deprecated; Core ML is model-based (too heavyweight)
//
// APPLE DECIDES SCHEDULING: Just like Core ML, Apple's runtime decides
// whether to use ANE based on:
//   - Operation type (matmul is ANE-friendly)
//   - Data size and shape (large batches preferred)
//   - Graph complexity (fused ops benefit more)
//   - Current system load
//   - Power/thermal constraints
//
// PERFORMANCE NOTES:
//   - For small matrices (<1024), Accelerate (CPU BLAS) is faster
//   - For large matrices (≥1024), Metal/ANE are 2x faster
//   - MPSGraph may use GPU instead of ANE for small workloads
//   - ANE shines with batch inference, not single operations
//
// ===========================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface ANEMatMulExecutor : NSObject {
    @public
    MPSGraph *graph;
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    MPSGraphTensor *inputATensor;
    MPSGraphTensor *inputBTensor;
    MPSGraphTensor *outputTensor;
    MPSGraphExecutable *executable;
    int m, n, k;
}

- (instancetype)initWithDevice:(id<MTLDevice>)dev m:(int)m_val n:(int)n_val k:(int)k_val;
- (void)executeWithA:(const float*)a b:(const float*)b output:(float*)c;

@end

@implementation ANEMatMulExecutor

- (instancetype)initWithDevice:(id<MTLDevice>)dev m:(int)m_val n:(int)n_val k:(int)k_val {
    self = [super init];
    if (self) {
        device = dev;
        commandQueue = [device newCommandQueue];
        m = m_val;
        n = n_val;
        k = k_val;

        // Create MPSGraph
        graph = [[MPSGraph alloc] init];

        // Define input placeholders
        // Note: MPSGraph uses NCHW layout by default, but for matmul we use simple 2D
        inputATensor = [graph placeholderWithShape:@[@(m), @(k)]
                                          dataType:MPSDataTypeFloat32
                                              name:@"inputA"];

        inputBTensor = [graph placeholderWithShape:@[@(k), @(n)]
                                          dataType:MPSDataTypeFloat32
                                              name:@"inputB"];

        // Create matrix multiplication operation
        outputTensor = [graph matrixMultiplicationWithPrimaryTensor:inputATensor
                                                    secondaryTensor:inputBTensor
                                                               name:@"output"];

        // Create MPSGraphDevice from MTLDevice
        MPSGraphDevice *graphDevice = [MPSGraphDevice deviceWithMTLDevice:device];

        // Compile the graph
        // This is where Apple decides CPU/GPU/ANE
        MPSGraphCompilationDescriptor *descriptor = [[MPSGraphCompilationDescriptor alloc] init];

        // Request ANE if available
        if (@available(macOS 14.0, iOS 17.0, *)) {
            descriptor.optimizationProfile = MPSGraphOptimizationProfilePerformance;
        }

        executable = [graph compileWithDevice:graphDevice
                                          feeds:@{inputATensor: [[MPSGraphShapedType alloc] initWithShape:@[@(m), @(k)] dataType:MPSDataTypeFloat32],
                                                  inputBTensor: [[MPSGraphShapedType alloc] initWithShape:@[@(k), @(n)] dataType:MPSDataTypeFloat32]}
                                       targetTensors:@[outputTensor]
                               targetOperations:nil
                             compilationDescriptor:descriptor];
    }
    return self;
}

- (void)executeWithA:(const float*)a b:(const float*)b output:(float*)c {
    @autoreleasepool {
        // Create Metal buffers
        size_t a_size = m * k * sizeof(float);
        size_t b_size = k * n * sizeof(float);
        size_t c_size = m * n * sizeof(float);

        id<MTLBuffer> bufferA = [device newBufferWithBytes:a
                                                    length:a_size
                                                   options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferB = [device newBufferWithBytes:b
                                                    length:b_size
                                                   options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferC = [device newBufferWithLength:c_size
                                                     options:MTLResourceStorageModeShared];

        // Create tensor data
        MPSGraphTensorData *tensorDataA = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bufferA
            shape:@[@(m), @(k)]
            dataType:MPSDataTypeFloat32];

        MPSGraphTensorData *tensorDataB = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bufferB
            shape:@[@(k), @(n)]
            dataType:MPSDataTypeFloat32];

        MPSGraphTensorData *tensorDataC = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bufferC
            shape:@[@(m), @(n)]
            dataType:MPSDataTypeFloat32];

        // Execute the graph
        NSArray *inputsArray = @[tensorDataA, tensorDataB];
        NSArray *resultsArray = @[tensorDataC];

        [executable runWithMTLCommandQueue:commandQueue
                               inputsArray:inputsArray
                              resultsArray:resultsArray
                       executionDescriptor:nil];

        // Wait for completion
        [commandQueue insertDebugCaptureBoundary];

        // Copy result back
        memcpy(c, bufferC.contents, c_size);
    }
}

@end

// C interface for Go
typedef void* ANEExecutorHandle;

ANEExecutorHandle ane_create_executor(int m, int n, int k) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return NULL;
        }

        ANEMatMulExecutor *executor = [[ANEMatMulExecutor alloc] initWithDevice:device
                                                                               m:m
                                                                               n:n
                                                                               k:k];
        return (__bridge_retained void*)executor;
    }
}

void ane_destroy_executor(ANEExecutorHandle handle) {
    if (handle) {
        ANEMatMulExecutor *executor = (__bridge_transfer ANEMatMulExecutor*)handle;
        executor = nil;
    }
}

int ane_execute_matmul(ANEExecutorHandle handle,
                       const float* a,
                       const float* b,
                       float* c) {
    @autoreleasepool {
        if (!handle) {
            return -1;
        }

        ANEMatMulExecutor *executor = (__bridge ANEMatMulExecutor*)handle;

        @try {
            [executor executeWithA:a b:b output:c];
            return 0;
        } @catch (NSException *exception) {
            NSLog(@"ANE execution failed: %@", exception);
            return -2;
        }
    }
}

const char* ane_get_info() {
    return "ANE via MPSGraph:\n"
           "- Graph-based compute framework\n"
           "- Can target CPU, GPU, or ANE\n"
           "- Apple runtime decides which accelerator to use\n"
           "- Prefer ANE for: matmul, conv, common ML ops\n"
           "- May use GPU for: large matrices, high precision\n"
           "- May use CPU for: small matrices, uncommon ops\n"
           "\n"
           "To verify ANE usage:\n"
           "1. Open Instruments.app\n"
           "2. Choose 'Logging' template\n"
           "3. Add 'os_signpost' instrument\n"
           "4. Filter for 'com.apple.mps'\n"
           "5. Run your code and check which accelerator is used";
}
