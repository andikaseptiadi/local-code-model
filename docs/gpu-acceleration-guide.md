# GPU Acceleration Guide

This guide covers how to leverage GPUs for transformer training and inference, mapping our pure Go optimizations to GPU-accelerated equivalents across different architectures (CUDA, Metal, ROCm).

## Table of Contents

1. [GPU Architecture Overview](#gpu-architecture-overview)
2. [CPU vs GPU: When to Use Each](#cpu-vs-gpu-when-to-use-each)
3. [CUDA (NVIDIA)](#cuda-nvidia)
4. [Metal (Apple Silicon)](#metal-apple-silicon)
5. [ROCm (AMD)](#rocm-amd)
6. [Mapping CPU Optimizations to GPU](#mapping-cpu-optimizations-to-gpu)
7. [Training on GPUs](#training-on-gpus)
8. [Inference on GPUs](#inference-on-gpus)
9. [Multi-GPU Training](#multi-gpu-training)
10. [Benchmarking Methodology](#benchmarking-methodology)

---

## GPU Architecture Overview

### Why GPUs for Deep Learning?

GPUs excel at transformer training/inference due to:

**Massive Parallelism:**
- CPUs: 8-64 cores (execute ~100 operations/cycle)
- GPUs: 1000-10000 cores (execute ~100,000 operations/cycle)

**Memory Bandwidth:**
- CPU: 50-100 GB/s (DDR4/DDR5)
- GPU: 500-2000 GB/s (HBM2/HBM3)

**Specialized Hardware:**
- **Tensor Cores** (NVIDIA): 4×4 matrix multiply-accumulate in one instruction
- **Matrix Engines** (Apple): Optimized for ML matrix operations
- **Matrix Cores** (AMD): Similar to Tensor Cores

### Hardware Comparison

| Hardware | Compute (TFLOPS FP32) | Memory Bandwidth | Sweet Spot |
|----------|----------------------|------------------|------------|
| M4 Max | 16 TF (GPU) + ANE | 546 GB/s | Inference, small training |
| RTX 5090 | 120 TF (Tensor Cores) | 1792 GB/s | Large-scale training |
| RTX 4090 | 82 TF | 1008 GB/s | Training & inference |
| AMD MI300X | 163 TF | 5300 GB/s | Datacenter training |
| AWS Graviton4 | 2 TF (CPU) | 256 GB/s | Inference at scale |

**Key Insight:** Memory bandwidth often matters more than raw compute for transformers (memory-bound workload).

---

## CPU vs GPU: When to Use Each

### Use CPU When:

✅ **Model is small** (<100M parameters)
- GPU overhead exceeds benefits
- Example: 6-layer transformer (30M params) trains faster on modern CPU

✅ **Batch size is 1** (single-sample inference)
- Can't exploit GPU parallelism
- CPU has lower latency (no PCIe transfer)

✅ **Sequence length is short** (<128 tokens)
- Insufficient work to saturate GPU
- CPU cache is effective

✅ **Development/debugging**
- Faster iteration cycle
- Better error messages
- Easier profiling

### Use GPU When:

✅ **Model is large** (>100M parameters)
- Amortizes GPU overhead
- Example: 12-layer transformer (120M params) trains 10-50× faster on GPU

✅ **Batch size is large** (≥8)
- Exploits massive parallelism
- Better memory bandwidth utilization

✅ **Sequence length is long** (>512 tokens)
- Matrix operations dominate
- GPU memory bandwidth advantage

✅ **Training for hours/days**
- GPU speedup compounds
- Cost-effective at scale

### Hybrid Approach:

For **inference serving**, consider:
```
CPU: Handle requests, tokenization, post-processing
GPU: Batched forward passes every 10-50ms
```

This maximizes throughput while minimizing latency.

---

## CUDA (NVIDIA)

### Architecture: RTX 5090

**Specs:**
- 21,760 CUDA cores
- 680 Tensor Cores (5th gen)
- 32 GB GDDR7 (1792 GB/s bandwidth)
- 120 TF FP32, 480 TF FP16 (Tensor Cores)

### CUDA Programming Model

```go
// Go wrapper for CUDA operations
package cuda

/*
#cgo LDFLAGS: -lcudart -lcublas
#include <cuda_runtime.h>
#include <cublas_v2.h>
*/
import "C"
import "unsafe"

// CUDADevice represents a GPU
type CUDADevice struct {
    deviceID int
    handle   C.cublasHandle_t
}

// Initialize CUDA
func NewCUDADevice(deviceID int) (*CUDADevice, error) {
    C.cudaSetDevice(C.int(deviceID))

    var handle C.cublasHandle_t
    status := C.cublasCreate(&handle)
    if status != 0 {
        return nil, fmt.Errorf("cuBLAS init failed: %d", status)
    }

    return &CUDADevice{
        deviceID: deviceID,
        handle:   handle,
    }, nil
}

// Matrix multiplication on GPU
func (d *CUDADevice) MatMul(A, B *Tensor) *Tensor {
    M, K := A.Shape()[0], A.Shape()[1]
    N := B.Shape()[1]

    // Allocate GPU memory
    var d_A, d_B, d_C unsafe.Pointer
    C.cudaMalloc(&d_A, C.size_t(M*K*4)) // 4 bytes/float32
    C.cudaMalloc(&d_B, C.size_t(K*N*4))
    C.cudaMalloc(&d_C, C.size_t(M*N*4))

    // Copy data to GPU
    C.cudaMemcpy(d_A, unsafe.Pointer(&A.Data[0]), C.size_t(M*K*4), C.cudaMemcpyHostToDevice)
    C.cudaMemcpy(d_B, unsafe.Pointer(&B.Data[0]), C.size_t(K*N*4), C.cudaMemcpyHostToDevice)

    // Perform matrix multiplication using cuBLAS
    alpha := C.float(1.0)
    beta := C.float(0.0)
    C.cublasSgemm(
        d.handle,
        C.CUBLAS_OP_N, C.CUBLAS_OP_N,
        C.int(N), C.int(M), C.int(K),
        &alpha,
        (*C.float)(d_B), C.int(N),
        (*C.float)(d_A), C.int(K),
        &beta,
        (*C.float)(d_C), C.int(N),
    )

    // Copy result back to CPU
    C_data := make([]float32, M*N)
    C.cudaMemcpy(unsafe.Pointer(&C_data[0]), d_C, C.size_t(M*N*4), C.cudaMemcpyDeviceToHost)

    // Free GPU memory
    C.cudaFree(d_A)
    C.cudaFree(d_B)
    C.cudaFree(d_C)

    return &Tensor{Data: C_data, shape: []int{M, N}}
}
```

### Tensor Cores (Mixed Precision)

Tensor Cores perform `D = A × B + C` in FP16/BF16 with FP32 accumulation:

```go
// Enable Tensor Cores via cuBLAS
func (d *CUDADevice) MatMulTensorCore(A, B *Tensor) *Tensor {
    // Convert to FP16
    A_fp16 := Float32ToFloat16(A)
    B_fp16 := Float32ToFloat16(B)

    // Allocate GPU memory
    var d_A, d_B, d_C unsafe.Pointer
    M, K := A.Shape()[0], A.Shape()[1]
    N := B.Shape()[1]

    C.cudaMalloc(&d_A, C.size_t(M*K*2)) // 2 bytes/fp16
    C.cudaMalloc(&d_B, C.size_t(K*N*2))
    C.cudaMalloc(&d_C, C.size_t(M*N*4)) // Result in FP32

    // Copy FP16 data to GPU
    C.cudaMemcpy(d_A, unsafe.Pointer(&A_fp16[0]), C.size_t(M*K*2), C.cudaMemcpyHostToDevice)
    C.cudaMemcpy(d_B, unsafe.Pointer(&B_fp16[0]), C.size_t(K*N*2), C.cudaMemcpyHostToDevice)

    // Use FP16 Tensor Core GEMM
    alpha := C.float(1.0)
    beta := C.float(0.0)
    C.cublasGemmEx(
        d.handle,
        C.CUBLAS_OP_N, C.CUBLAS_OP_N,
        C.int(N), C.int(M), C.int(K),
        &alpha,
        d_B, C.CUDA_R_16F, C.int(N),      // FP16 input
        d_A, C.CUDA_R_16F, C.int(K),      // FP16 input
        &beta,
        d_C, C.CUDA_R_32F, C.int(N),      // FP32 output
        C.CUDA_R_32F,                     // FP32 compute
        C.CUBLAS_GEMM_DEFAULT_TENSOR_OP,  // Enable Tensor Cores
    )

    // Copy FP32 result back
    C_data := make([]float32, M*N)
    C.cudaMemcpy(unsafe.Pointer(&C_data[0]), d_C, C.size_t(M*N*4), C.cudaMemcpyDeviceToHost)

    C.cudaFree(d_A)
    C.cudaFree(d_B)
    C.cudaFree(d_C)

    return &Tensor{Data: C_data, shape: []int{M, N}}
}
```

**Performance gain:** 4-8× faster than FP32 on Tensor Cores

### Flash Attention on CUDA

```go
// Flash Attention kernel (simplified)
/*
__global__ void flash_attention_kernel(
    const half* Q,    // [batch, heads, seq_len, head_dim]
    const half* K,
    const half* V,
    half* O,          // Output
    int seq_len,
    int head_dim,
    int block_size
) {
    // Shared memory for tile (L1 cache)
    __shared__ half Q_tile[BLOCK_SIZE][HEAD_DIM];
    __shared__ half K_tile[BLOCK_SIZE][HEAD_DIM];
    __shared__ half V_tile[BLOCK_SIZE][HEAD_DIM];

    // Each block processes one query tile
    int q_start = blockIdx.x * BLOCK_SIZE;

    // Load Q tile to shared memory
    load_tile(Q, Q_tile, q_start, head_dim);

    // Initialize output accumulators
    float O_acc[HEAD_DIM] = {0};
    float m = -INFINITY;  // Max value for numerical stability
    float l = 0.0f;       // Sum of exponentials

    // Iterate over K/V tiles
    for (int k_start = 0; k_start < seq_len; k_start += BLOCK_SIZE) {
        // Load K/V tiles
        load_tile(K, K_tile, k_start, head_dim);
        load_tile(V, V_tile, k_start, head_dim);

        // Compute Q @ K^T for this tile
        float scores[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            scores[i] = dot_product(Q_tile[threadIdx.x], K_tile[i], head_dim);
            scores[i] /= sqrtf(head_dim);
        }

        // Online softmax (numerically stable)
        float m_new = max(m, max_array(scores, BLOCK_SIZE));
        float l_correction = expf(m - m_new);

        // Update output with corrected exponentials
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float exp_score = expf(scores[i] - m_new);
            for (int d = 0; d < HEAD_DIM; d++) {
                O_acc[d] = O_acc[d] * l_correction + exp_score * V_tile[i][d];
            }
            l = l * l_correction + exp_score;
        }

        m = m_new;
    }

    // Normalize output
    for (int d = 0; d < HEAD_DIM; d++) {
        O[blockIdx.x * BLOCK_SIZE * HEAD_DIM + threadIdx.x * HEAD_DIM + d] = O_acc[d] / l;
    }
}
*/
import "C"

func (d *CUDADevice) FlashAttention(Q, K, V *Tensor, blockSize int) *Tensor {
    // Launch Flash Attention kernel
    // ... CUDA kernel launch code ...
    return output
}
```

**Memory savings:** 10-100× less HBM traffic compared to standard attention

---

## Metal (Apple Silicon)

### Architecture: M4 Max

**Specs:**
- 40-core GPU (1280 ALUs)
- 16-core Neural Engine (38 TOPS)
- 128 GB unified memory (546 GB/s bandwidth)
- Hardware-accelerated matrix operations

**Unified Memory Advantage:** Zero-copy between CPU and GPU (no PCIe bottleneck)

### Metal Performance Shaders (MPS)

```go
// Go wrapper for Metal operations
package metal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

id<MTLDevice> getDevice() {
    return MTLCreateSystemDefaultDevice();
}

id<MTLCommandQueue> createCommandQueue(id<MTLDevice> device) {
    return [device newCommandQueue];
}
*/
import "C"

type MetalDevice struct {
    device       C.id
    commandQueue C.id
}

func NewMetalDevice() *MetalDevice {
    device := C.getDevice()
    return &MetalDevice{
        device:       device,
        commandQueue: C.createCommandQueue(device),
    }
}

// Matrix multiplication using MPS
func (m *MetalDevice) MatMul(A, B *Tensor) *Tensor {
    // Create Metal buffers (zero-copy with unified memory)
    // metalA := createBuffer(m.device, A.Data)
    // metalB := createBuffer(m.device, B.Data)
    // metalC := createBuffer(m.device, outputSize)

    // Create MPS matrix multiplication descriptor
    /*
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
        columns:K
        rowBytes:K*sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
        transposeLeft:NO
        transposeRight:NO
        resultRows:M
        resultColumns:N
        interiorColumns:K
        alpha:1.0
        beta:0.0];

    [matmul encodeToCommandBuffer:commandBuffer
        leftMatrix:matrixA
        rightMatrix:matrixB
        resultMatrix:matrixC];
    */

    return output
}
```

### Apple Neural Engine (ANE)

The ANE is optimized for:
- **Inference** (not training - no backprop support)
- **FP16/INT8** operations
- **Batch size 1** (single-sample inference)

```go
// Use Core ML for ANE acceleration
package coreml

/*
#cgo LDFLAGS: -framework CoreML
#import <CoreML/CoreML.h>

// Export transformer model to Core ML format
// (Requires converting from Go to ONNX to Core ML)
*/

func ExportToCoreML(model *Transformer, outputPath string) error {
    // 1. Export model to ONNX format
    onnxModel := ExportToONNX(model)

    // 2. Convert ONNX to Core ML (using Python coremltools)
    // $ python3 -m coremltools.converters.onnx convert model.onnx -o model.mlmodel

    // 3. Load Core ML model
    // The ANE will automatically be used for compatible operations

    return nil
}

// Inference using ANE
func (m *CoreMLModel) Forward(input *Tensor) *Tensor {
    // Core ML automatically dispatches to ANE when possible
    // Operations that run on ANE: MatMul, Conv, LayerNorm, GELU, Softmax
    // Operations that run on GPU: Custom ops, large batch sizes
    return m.predict(input)
}
```

**ANE Performance:**
- 5-10× faster than GPU for single-sample inference
- Uses <1W power (vs 20W GPU)
- Best for: ChatGPT-style autoregressive generation

### Metal Shader Optimization

```metal
// Custom Metal shader for fused GELU
kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    // GELU: x * Φ(x) ≈ x * σ(1.702 * x)
    float sigmoid = 1.0f / (1.0f + exp(-1.702f * x));
    output[id] = x * sigmoid;
}
```

**Fusion benefits:** 2-3× faster than separate operations (reduces memory bandwidth)

---

## ROCm (AMD)

### Architecture: AMD MI300X (Datacenter)

**Specs:**
- 19,456 stream processors
- 192 GB HBM3 (5.3 TB/s bandwidth)
- 163 TF FP32, 1.3 PF FP16

### ROCm Programming Model

ROCm uses HIP (similar to CUDA):

```go
// Go wrapper for ROCm/HIP
package rocm

/*
#cgo LDFLAGS: -lhip -lrocblas
#include <hip/hip_runtime.h>
#include <rocblas.h>
*/
import "C"

type ROCmDevice struct {
    deviceID int
    handle   C.rocblas_handle
}

func NewROCmDevice(deviceID int) (*ROCmDevice, error) {
    C.hipSetDevice(C.int(deviceID))

    var handle C.rocblas_handle
    C.rocblas_create_handle(&handle)

    return &ROCmDevice{
        deviceID: deviceID,
        handle:   handle,
    }, nil
}

// Matrix multiplication using rocBLAS
func (d *ROCmDevice) MatMul(A, B *Tensor) *Tensor {
    // Very similar to CUDA/cuBLAS implementation
    // hipMalloc instead of cudaMalloc
    // hipMemcpy instead of cudaMemcpy
    // rocblas_sgemm instead of cublasSgemm

    return output
}
```

**Note:** ROCm API is nearly identical to CUDA (designed for portability)

---

## Mapping CPU Optimizations to GPU

Our pure Go optimizations map to GPU equivalents:

| CPU Optimization | GPU Equivalent | Speedup |
|------------------|----------------|---------|
| Parallel MatMul | cuBLAS/MPS MatMul | 10-50× |
| Blocked MatMul | Automatic (GPU caches) | Built-in |
| SIMD (AVX/NEON) | CUDA cores/GPU ALUs | 100× |
| Mixed Precision | Tensor Cores/ANE | 4-8× |
| Flash Attention | Flash Attention kernel | 2-4× |
| KV Cache | Same (CPU→GPU memory) | 50× |
| Gradient Checkpointing | Same (recompute on GPU) | Same ratio |

### Hybrid CPU-GPU Pipeline

```go
type HybridModel struct {
    cpu *Transformer
    gpu *CUDADevice
}

func (h *HybridModel) Forward(input []int) *Tensor {
    // Tokenization on CPU (fast, no benefit from GPU)
    tokens := h.tokenize(input)

    // Transfer threshold: only use GPU if model is large enough
    if h.cpu.Config.NumLayers > 6 && len(tokens) > 128 {
        // Transfer to GPU
        gpuTokens := h.gpu.Upload(tokens)

        // Forward pass on GPU
        output := h.gpu.Forward(gpuTokens)

        // Transfer back to CPU
        return h.gpu.Download(output)
    } else {
        // Forward pass on CPU (faster for small models)
        return h.cpu.Forward(tokens)
    }
}
```

---

## Training on GPUs

### Single-GPU Training

```go
func TrainOnGPU(model *Transformer, data [][]int, device *CUDADevice) {
    // Upload model to GPU
    gpuModel := device.UploadModel(model)

    for epoch := 0; epoch < numEpochs; epoch++ {
        for _, batch := range data {
            // Upload batch to GPU
            gpuBatch := device.Upload(batch)

            // Forward pass (on GPU)
            logits := gpuModel.Forward(gpuBatch)

            // Compute loss (on GPU)
            loss := device.ComputeLoss(logits, gpuBatch)

            // Backward pass (on GPU)
            gpuModel.Backward(loss)

            // Update weights (on GPU)
            device.OptimizerStep(gpuModel)

            // Only transfer loss back to CPU for logging
            cpuLoss := device.Download(loss)
            fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, cpuLoss)
        }
    }
}
```

**Key principle:** Minimize CPU↔GPU transfers (bottleneck)

### Memory Management

```go
// GPU memory is limited - use gradient checkpointing for large models
func TrainLargeModelOnGPU(model *Transformer, device *CUDADevice) {
    config := &CheckpointConfig{
        Enabled:          true,
        CheckpointEveryN: 4,  // More aggressive than CPU (limited GPU memory)
    }

    // With checkpointing, can train 2-4× larger models on same GPU
    for epoch := 0; epoch < numEpochs; epoch++ {
        for _, batch := range batches {
            loss := model.ForwardWithCheckpointing(batch, config)
            model.Backward(loss)
        }
    }
}
```

---

## Inference on GPUs

### Batched Inference (Throughput)

```go
// Maximize GPU utilization with large batches
func BatchedInferenceGPU(model *Transformer, prompts [][]int, device *CUDADevice) [][]int {
    batchSize := 32  // Large batch to saturate GPU

    results := make([][]int, len(prompts))

    for i := 0; i < len(prompts); i += batchSize {
        end := min(i+batchSize, len(prompts))
        batch := prompts[i:end]

        // Upload batch to GPU
        gpuBatch := device.Upload(batch)

        // Generate for entire batch simultaneously
        gpuOutputs := device.GenerateBatch(gpuBatch, maxLen)

        // Download results
        results[i:end] = device.Download(gpuOutputs)
    }

    return results
}
```

**Throughput:** 1000-10000 tokens/second (vs 10-100 on CPU)

### Low-Latency Inference (Single Sample)

```go
// For single-sample inference, use ANE (Apple) or CPU (NVIDIA)
func LowLatencyInference(model *Transformer, prompt []int) []int {
    // On Apple Silicon: Use ANE via Core ML
    if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
        coreMLModel := ConvertToCoreML(model)
        return coreMLModel.Generate(prompt, maxLen)
    }

    // On NVIDIA: CPU might be faster for batch=1 (no transfer overhead)
    if batchSize == 1 && modelSize < 1000000000 {  // <1B params
        return model.Generate(prompt, maxLen)  // CPU
    }

    // Otherwise use GPU
    return gpuModel.Generate(prompt, maxLen)
}
```

**Latency:** 5-50ms per token (ANE/CPU) vs 10-100ms (NVIDIA GPU for batch=1)

---

## Multi-GPU Training

### Data Parallelism

```go
// Replicate model on each GPU, split data across GPUs
func DataParallelTraining(model *Transformer, data [][]int, devices []*CUDADevice) {
    // Replicate model to all GPUs
    gpuModels := make([]*GPUModel, len(devices))
    for i, device := range devices {
        gpuModels[i] = device.UploadModel(model)
    }

    for epoch := 0; epoch < numEpochs; epoch++ {
        // Split batch across GPUs
        batchesPerGPU := len(data) / len(devices)

        var wg sync.WaitGroup
        for i, gpuModel := range gpuModels {
            wg.Add(1)
            go func(idx int, gm *GPUModel) {
                defer wg.Done()

                // Each GPU processes its subset
                start := idx * batchesPerGPU
                end := start + batchesPerGPU

                for _, batch := range data[start:end] {
                    loss := gm.Forward(batch)
                    gm.Backward(loss)
                }
            }(i, gpuModel)
        }
        wg.Wait()

        // Synchronize gradients across GPUs (all-reduce)
        SynchronizeGradients(gpuModels, devices)

        // Update weights on all GPUs
        for _, gpuModel := range gpuModels {
            gpuModel.OptimizerStep()
        }
    }
}

// All-reduce: Average gradients across GPUs
func SynchronizeGradients(models []*GPUModel, devices []*CUDADevice) {
    // Use NCCL (NVIDIA) or RCCL (AMD) for efficient multi-GPU communication
    // For simplicity, shown as pseudo-code:

    for _, param := range models[0].Parameters() {
        // Gather gradients from all GPUs
        grads := make([]Tensor, len(models))
        for i, model := range models {
            grads[i] = model.GetGradient(param.Name)
        }

        // Average gradients
        avgGrad := AverageGradients(grads)

        // Broadcast averaged gradient back to all GPUs
        for _, model := range models {
            model.SetGradient(param.Name, avgGrad)
        }
    }
}
```

**Scaling:** Near-linear up to 4-8 GPUs (communication overhead grows)

### Model Parallelism (for very large models)

```go
// Split model across GPUs (different layers on different GPUs)
func ModelParallelTraining(model *Transformer, data [][]int, devices []*CUDADevice) {
    // Split layers across GPUs
    layersPerGPU := model.Config.NumLayers / len(devices)

    for epoch := 0; epoch < numEpochs; epoch++ {
        for _, batch := range data {
            // Forward pass: pipeline through GPUs
            x := batch
            for i, device := range devices {
                startLayer := i * layersPerGPU
                endLayer := startLayer + layersPerGPU

                // Upload input to GPU i
                gpuX := device.Upload(x)

                // Process layers on this GPU
                gpuX = device.ForwardLayers(gpuX, startLayer, endLayer)

                // Download for next GPU (pipeline bubble - inefficient!)
                x = device.Download(gpuX)
            }

            // Backward pass: reverse pipeline
            // ... (similar reverse flow)
        }
    }
}
```

**Note:** Requires pipeline parallelism (GPipe, PipeDream) to be efficient

---

## Benchmarking Methodology

### Measuring GPU Utilization

```bash
# NVIDIA
nvidia-smi dmon -s u  # Monitor GPU utilization

# Expected for training: 80-100% GPU utilization
# If <50%: CPU bottleneck (data loading, preprocessing)

# AMD
rocm-smi -u  # Monitor GPU utilization

# Apple
sudo powermetrics --samplers gpu_power  # Monitor GPU power (proxy for utilization)
```

### Profiling GPU Operations

```go
// CUDA profiling
import "github.com/NVIDIA/cuda-toolkit/nvtx"

func ProfiledForward(model *GPUModel, input *Tensor) *Tensor {
    nvtx.RangePush("forward_pass")
    defer nvtx.RangePop()

    nvtx.RangePush("embedding")
    x := model.Embedding(input)
    nvtx.RangePop()

    for i, layer := range model.Layers {
        nvtx.RangePush(fmt.Sprintf("layer_%d", i))
        x = layer.Forward(x)
        nvtx.RangePop()
    }

    return x
}

// View profile:
// $ nsys profile ./train
// $ nsys-ui profile.qdrep
```

### Benchmark Results (Expected)

**Training Speed (tokens/second):**

| Model Size | CPU (M4 Max) | GPU (RTX 5090) | Speedup |
|------------|-------------|----------------|---------|
| 30M params | 5,000 | 15,000 | 3× |
| 120M params | 800 | 40,000 | 50× |
| 350M params | 150 | 25,000 | 167× |
| 1B params | OOM | 8,000 | ∞ |

**Inference Latency (ms/token, batch=1):**

| Model Size | CPU | ANE (M4) | GPU (RTX 5090) |
|------------|-----|----------|----------------|
| 30M params | 10 | 2 | 15 |
| 120M params | 50 | 8 | 20 |
| 350M params | 200 | 25 | 30 |
| 1B params | OOM | 80 | 40 |

**Key insight:** ANE wins for inference latency, GPU wins for training/throughput

---

## Practical Recommendations

### For Your Hardware:

**M4 Max (Mac):**
- ✅ Use for inference (ANE acceleration)
- ✅ Use for small model training (<100M params)
- ❌ Avoid for large model training (limited memory bandwidth)

**RTX 5090:**
- ✅ Use for all training (10-100× faster than CPU)
- ✅ Use for high-throughput inference (large batches)
- ❌ Overkill for single-sample inference (use CPU/ANE instead)

### Next Steps:

1. **Implement CUDA backend** for MatMul, Softmax, LayerNorm
2. **Benchmark training** on both systems (M4 Max vs RTX 5090)
3. **Profile GPU utilization** (identify bottlenecks)
4. **Implement Metal backend** for Apple Silicon optimization
5. **Create multi-GPU training** script (if you have multiple GPUs)

Would you like me to implement any of these GPU backends?
