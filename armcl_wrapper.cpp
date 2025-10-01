// +build linux,arm64,cgo

// ARM Compute Library C++ Wrapper
// Provides C interface for Go CGo bindings

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/Tensor.h>
#include <cstring>
#include <iostream>

extern "C" {

// Opaque types
struct ACLContext {
    // Could hold global state if needed
    bool initialized;
};

struct ACLTensorWrapper {
    arm_compute::Tensor tensor;
};

struct ACLGEMMWrapper {
    arm_compute::NEGEMM gemm;
    bool configured;
};

// Initialize ARM Compute Library
void* acl_init(void) {
    ACLContext* ctx = new ACLContext();
    ctx->initialized = true;
    return ctx;
}

// Create tensor
void* acl_create_tensor(void* context, int rows, int cols) {
    if (!context) return nullptr;

    ACLTensorWrapper* wrapper = new ACLTensorWrapper();

    // Create tensor info for FP64
    arm_compute::TensorInfo info(
        arm_compute::TensorShape(cols, rows),  // ACL uses (width, height)
        1,  // num_channels
        arm_compute::DataType::F64
    );

    wrapper->tensor.allocator()->init(info);
    wrapper->tensor.allocator()->allocate();

    return wrapper;
}

// Set tensor data
void acl_set_tensor_data(void* tensor, double* data, int size) {
    if (!tensor || !data) return;

    ACLTensorWrapper* wrapper = static_cast<ACLTensorWrapper*>(tensor);

    // Map tensor buffer
    wrapper->tensor.map();

    // Copy data
    double* buffer = reinterpret_cast<double*>(wrapper->tensor.buffer());
    std::memcpy(buffer, data, size * sizeof(double));

    wrapper->tensor.unmap();
}

// Get tensor data
void acl_get_tensor_data(void* tensor, double* data, int size) {
    if (!tensor || !data) return;

    ACLTensorWrapper* wrapper = static_cast<ACLTensorWrapper*>(tensor);

    // Map tensor buffer
    wrapper->tensor.map();

    // Copy data
    const double* buffer = reinterpret_cast<const double*>(wrapper->tensor.buffer());
    std::memcpy(data, buffer, size * sizeof(double));

    wrapper->tensor.unmap();
}

// Create GEMM operation
void* acl_create_gemm(void* context, void* a, void* b, void* c) {
    if (!context || !a || !b || !c) return nullptr;

    ACLTensorWrapper* tensor_a = static_cast<ACLTensorWrapper*>(a);
    ACLTensorWrapper* tensor_b = static_cast<ACLTensorWrapper*>(b);
    ACLTensorWrapper* tensor_c = static_cast<ACLTensorWrapper*>(c);

    ACLGEMMWrapper* gemm = new ACLGEMMWrapper();

    // Configure GEMM: C = A * B
    // alpha=1.0, beta=0.0 (no accumulation)
    gemm->gemm.configure(
        &tensor_a->tensor,
        &tensor_b->tensor,
        nullptr,  // No bias
        &tensor_c->tensor,
        1.0f,  // alpha
        0.0f   // beta
    );

    gemm->configured = true;

    return gemm;
}

// Run GEMM
void acl_run_gemm(void* gemm) {
    if (!gemm) return;

    ACLGEMMWrapper* wrapper = static_cast<ACLGEMMWrapper*>(gemm);

    if (wrapper->configured) {
        wrapper->gemm.run();
    }
}

// Destroy tensor
void acl_destroy_tensor(void* tensor) {
    if (tensor) {
        ACLTensorWrapper* wrapper = static_cast<ACLTensorWrapper*>(tensor);
        delete wrapper;
    }
}

// Destroy GEMM
void acl_destroy_gemm(void* gemm) {
    if (gemm) {
        ACLGEMMWrapper* wrapper = static_cast<ACLGEMMWrapper*>(gemm);
        delete wrapper;
    }
}

// Destroy context
void acl_destroy_context(void* context) {
    if (context) {
        ACLContext* ctx = static_cast<ACLContext*>(context);
        delete ctx;
    }
}

// Check availability
int acl_available(void) {
    // If this compiles and links, ACL is available
    return 1;
}

// Get info
const char* acl_get_info(void) {
    return "ARM Compute Library v24.x - Optimized for ARM Neoverse and Cortex-A";
}

} // extern "C"
