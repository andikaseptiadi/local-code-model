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

## 6. AWS Trainium & Inferentia (Custom AI Accelerators)

Amazon's custom-designed chips for machine learning: **Trainium** (training) and **Inferentia** (inference). These are purpose-built alternatives to NVIDIA GPUs in AWS, offering better cost-performance for specific workloads.

### Overview

| Chip | Purpose | Programming Model | Key Features |
|------|---------|------------------|--------------|
| **Trainium** | Training | AWS Neuron SDK | 16 NeuronCores per chip, FP32/BF16/FP16, collective communication |
| **Inferentia2** | Inference | AWS Neuron SDK | 2 NeuronCores per chip, INT8/FP16/BF16, ultra-low latency |

**Why Trainium/Inferentia?**
- Cost: 50% cheaper than comparable NVIDIA instances (per TFLOP)
- Integration: Deep AWS ecosystem integration (SageMaker, EKS, EC2)
- Efficiency: Purpose-built for transformers (optimized for attention, embeddings)
- Availability: No GPU shortage concerns in AWS regions

### Architecture Deep Dive

**NeuronCore:**
- Tensor compute engine (matrix multiply, activation functions)
- Vector/scalar compute engines
- On-chip SRAM (24 MB per NeuronCore on Trainium)
- Direct inter-chip connectivity (NeuronLink)

**Trainium (trn1 instances):**
- trn1.2xlarge: 1 Trainium chip (16 NeuronCores)
- trn1.32xlarge: 16 Trainium chips (256 NeuronCores), 512 GB HBM

**Inferentia2 (inf2 instances):**
- inf2.xlarge: 1 Inferentia2 chip (2 NeuronCores)
- inf2.48xlarge: 12 Inferentia2 chips (24 NeuronCores)

### Go Integration via AWS Neuron SDK

AWS Neuron SDK provides C++ APIs that we can wrap with CGO:

```go
package neuron

/*
#cgo CFLAGS: -I/opt/aws/neuron/include
#cgo LDFLAGS: -L/opt/aws/neuron/lib -lnrt -lnccom
#include <nrt/nrt.h>
#include <nccom/nccom.h>

// Wrapper for initializing Neuron runtime
int init_neuron() {
    return nrt_init(NRT_FRAMEWORK_TYPE_NO_FW);
}

// Wrapper for allocating tensor on NeuronCore
void* alloc_neuron_tensor(size_t size) {
    void* tensor;
    nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, size, &tensor);
    return tensor;
}
*/
import "C"
import "unsafe"

// NeuronDevice represents a Trainium or Inferentia device
type NeuronDevice struct {
    deviceID    int
    numCores    int
    runtimeHandle C.nrt_context_t
}

// NewNeuronDevice initializes a Neuron device
func NewNeuronDevice(deviceID int) (*NeuronDevice, error) {
    if C.init_neuron() != 0 {
        return nil, fmt.Errorf("failed to initialize Neuron runtime")
    }

    return &NeuronDevice{
        deviceID: deviceID,
        numCores: 16, // Trainium has 16 NeuronCores
    }, nil
}

// MatMul performs matrix multiplication on Trainium/Inferentia
func (d *NeuronDevice) MatMul(A, B *Tensor) *Tensor {
    M, K := A.Shape()[0], A.Shape()[1]
    N := B.Shape()[1]

    // Allocate tensors on NeuronCore SRAM
    d_A := C.alloc_neuron_tensor(C.size_t(M * K * 4))
    d_B := C.alloc_neuron_tensor(C.size_t(K * N * 4))
    d_C := C.alloc_neuron_tensor(C.size_t(M * N * 4))

    // Copy data to device (host → NeuronCore)
    C.nrt_tensor_write(d_A, unsafe.Pointer(&A.Data[0]), C.size_t(M*K*4))
    C.nrt_tensor_write(d_B, unsafe.Pointer(&B.Data[0]), C.size_t(K*N*4))

    // Execute MatMul on NeuronCore tensor engine
    // Note: In practice, you'd compile your model with neuron-cc first
    C.nrt_execute_gemm(d.runtimeHandle, d_A, d_B, d_C,
        C.int(M), C.int(N), C.int(K))

    // Copy result back
    C_data := make([]float32, M*N)
    C.nrt_tensor_read(d_C, unsafe.Pointer(&C_data[0]), C.size_t(M*N*4))

    // Free device memory
    C.nrt_tensor_free(d_A)
    C.nrt_tensor_free(d_B)
    C.nrt_tensor_free(d_C)

    return &Tensor{Data: C_data, shape: []int{M, N}}
}
```

### Compiling Models for Trainium/Inferentia

AWS Neuron requires ahead-of-time (AOT) compilation. You define your model in PyTorch/TensorFlow, then compile to a Neuron Executable File Format (NEFF):

```bash
# Install Neuron compiler
pip install torch-neuronx neuronx-cc

# Compile transformer for Inferentia2
python compile_for_neuron.py
```

**compile_for_neuron.py:**
```python
import torch
import torch_neuronx

# Define or load your transformer model
model = YourTransformer()
example_input = torch.randint(0, 1000, (1, 128))  # batch=1, seq_len=128

# Trace and compile for NeuronCore
model_neuron = torch_neuronx.trace(
    model,
    example_input,
    compiler_args="--target=inf2",  # or trn1 for Trainium
)

# Save compiled model
model_neuron.save("transformer_neuron.pt")
```

**Load and run from Go:**
```go
// Load pre-compiled NEFF model
func (d *NeuronDevice) LoadModel(path string) error {
    cPath := C.CString(path)
    defer C.free(unsafe.Pointer(cPath))

    ret := C.nrt_load_model(d.runtimeHandle, cPath)
    if ret != 0 {
        return fmt.Errorf("failed to load NEFF model")
    }
    return nil
}

// Run inference with compiled model
func (d *NeuronDevice) Forward(input []int) *Tensor {
    // Input tensor (batch × seq_len)
    inputData := make([]float32, len(input))
    for i, token := range input {
        inputData[i] = float32(token)
    }

    // Allocate and copy input
    d_input := C.alloc_neuron_tensor(C.size_t(len(input) * 4))
    C.nrt_tensor_write(d_input, unsafe.Pointer(&inputData[0]),
        C.size_t(len(input)*4))

    // Execute compiled model
    var d_output unsafe.Pointer
    C.nrt_execute_model(d.runtimeHandle, d_input, &d_output)

    // Read output logits
    outputSize := len(input) * vocabSize
    outputData := make([]float32, outputSize)
    C.nrt_tensor_read(d_output, unsafe.Pointer(&outputData[0]),
        C.size_t(outputSize*4))

    return &Tensor{Data: outputData, shape: []int{len(input), vocabSize}}
}
```

### Training on Trainium (trn1 instances)

Trainium is optimized for distributed training with NeuronLink (inter-chip communication):

```go
// Multi-chip data parallel training
type TrainiumCluster struct {
    chips []*NeuronDevice
    numChips int
}

func NewTrainiumCluster(numChips int) *TrainiumCluster {
    cluster := &TrainiumCluster{
        chips: make([]*NeuronDevice, numChips),
        numChips: numChips,
    }

    // Initialize all Trainium chips
    for i := 0; i < numChips; i++ {
        cluster.chips[i], _ = NewNeuronDevice(i)
    }

    // Initialize collective communication (all-reduce)
    C.nccom_init_all_reduce(C.int(numChips))

    return cluster
}

// Data parallel training step
func (c *TrainiumCluster) TrainStep(batches []*Batch) {
    // Distribute batches across chips
    for i, chip := range c.chips {
        go func(chipID int, batch *Batch) {
            // Forward pass on this chip
            logits := chip.Forward(batch.Input)
            loss := computeLoss(logits, batch.Target)

            // Backward pass (compute gradients)
            grads := chip.Backward(loss)

            // All-reduce gradients across chips (via NeuronLink)
            C.nccom_all_reduce(grads.devicePtr, C.int(chipID))

            // Update weights with averaged gradients
            chip.UpdateWeights(grads)
        }(i, batches[i])
    }
}
```

### Inference on Inferentia2 (inf2 instances)

Inferentia2 is optimized for ultra-low latency inference:

```go
// Batch inference on Inferentia2
func (d *NeuronDevice) BatchInference(inputs [][]int) []*Tensor {
    batchSize := len(inputs)
    results := make([]*Tensor, batchSize)

    // Inferentia2 supports batching for throughput
    for i, input := range inputs {
        results[i] = d.Forward(input)
    }

    return results
}

// Streaming inference (low latency)
func (d *NeuronDevice) StreamingInference(input []int, maxTokens int) []int {
    generated := make([]int, 0, maxTokens)
    current := input

    for len(generated) < maxTokens {
        // Run inference on Inferentia2
        logits := d.Forward(current)

        // Sample next token
        nextToken := sampleToken(logits)
        generated = append(generated, nextToken)

        // Append for next iteration (autoregressive)
        current = append(current, nextToken)
    }

    return generated
}
```

### Performance Characteristics

**Trainium (trn1.32xlarge):**
- Training throughput: ~90% of A100 at 50% cost
- BF16 performance: 840 TFLOPS (vs A100: 624 TFLOPS)
- Memory bandwidth: 820 GB/s per chip
- Sweet spot: Large transformer training (BERT, GPT-2, T5)

**Inferentia2 (inf2.48xlarge):**
- Inference latency: 3-5ms per token (comparable to A10G)
- Throughput: 10,000 inferences/sec with batching
- Cost: ~70% cheaper than GPU instances
- Sweet spot: Production inference at scale

**Comparison:**

| Workload | CPU | GPU (RTX 5090) | Trainium | Inferentia2 |
|----------|-----|----------------|----------|-------------|
| Training (1B params) | 72 hr | 8 hr | 10 hr | N/A |
| Training cost | $300 | $50 | $30 | N/A |
| Inference latency (ms) | 100 | 5 | 15 | 4 |
| Inference cost (per 1M tokens) | $2 | $0.50 | N/A | $0.15 |

### When to Use Trainium/Inferentia

**Use Trainium when:**
- ✅ Training large models (>1B params) in AWS
- ✅ Cost is a primary concern (50% cheaper than GPU)
- ✅ You can use PyTorch/TensorFlow with Neuron SDK
- ❌ Avoid for: Small models (<100M params), non-AWS deployments

**Use Inferentia2 when:**
- ✅ Production inference at scale in AWS
- ✅ Latency <10ms is acceptable
- ✅ Cost optimization for inference ($0.15/1M tokens)
- ❌ Avoid for: Lowest possible latency (<3ms), dynamic model changes

**Use GPU (CUDA) when:**
- ✅ Need lowest latency (<3ms)
- ✅ Rapid iteration/debugging (no AOT compilation)
- ✅ On-premises or multi-cloud deployments
- ✅ Small-scale training (<10 GPUs)

### Practical AWS Setup

**Launch trn1 instance for training:**
```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Deep Learning AMI
  --instance-type trn1.32xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx

# SSH and install Neuron SDK
ssh -i your-key.pem ubuntu@<instance-ip>
sudo apt install aws-neuronx-tools
```

**Deploy model to Inferentia2:**
```bash
# Compile model (on CPU instance, cheaper)
python compile_for_neuron.py --target=inf2

# Deploy to inf2 instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type inf2.xlarge

# Copy compiled model and run
scp transformer_neuron.pt ubuntu@<inf2-ip>:/models/
ssh ubuntu@<inf2-ip> ./serve_model
```

### Cost Analysis (AWS us-east-1, Jan 2025)

| Instance | Hourly Cost | Use Case | Cost per 1B token training |
|----------|-------------|----------|---------------------------|
| p4d.24xlarge (8× A100) | $32.77 | GPU training | $65 |
| trn1.32xlarge (16× Trainium) | $21.50 | Trainium training | $32 |
| g5.48xlarge (8× A10G) | $16.29 | GPU inference | N/A |
| inf2.48xlarge (12× Inferentia2) | $4.72 | Inferentia inference | N/A |

**Inference cost per 1M tokens:**
- g5.xlarge (A10G): $0.50
- inf2.xlarge (Inferentia2): $0.15 (70% cheaper)

---

## 7. Google TPU (Tensor Processing Units)

Google's custom ASICs for machine learning, designed specifically for TensorFlow and JAX. TPUs have been powering Google's AI infrastructure since 2016 (AlphaGo, Google Search, Gmail, etc.).

### Overview

| Generation | Compute (BF16) | Memory | Key Features |
|------------|---------------|---------|--------------|
| **TPU v4** | 275 TFLOPS | 32 GB HBM2 | 2x faster than v3, OCS networking |
| **TPU v5e** | 197 TFLOPS | 16 GB HBM2 | Cost-optimized for training/inference |
| **TPU v5p** | 459 TFLOPS | 95 GB HBM2 | Highest performance, large models |

**Why TPUs?**
- Performance: Optimized for matrix operations (MXU: Matrix Multiply Unit)
- Integration: Seamless TensorFlow/JAX integration in Google Cloud
- Pods: Scale to thousands of chips with ultra-fast interconnect (ICI: Inter-Chip Interconnect)
- Cost: Competitive pricing for large-scale training

### Architecture Deep Dive

**TPU Architecture:**
- MXU (Matrix Multiply Unit): 128×128 systolic array for BF16
- Vector Processing Unit (VPU): Element-wise operations (activation, softmax)
- Scalar Unit: Control flow, address generation
- High Bandwidth Memory (HBM): 900 GB/s bandwidth per chip
- Inter-Chip Interconnect (ICI): 4.8 Tbps per chip (v5p)

**Key Difference from GPUs:**
- TPUs: Fixed-function systolic arrays (ultra-efficient for matmul)
- GPUs: Programmable CUDA cores (flexible for any workload)

### Go Integration via XLA (Accelerated Linear Algebra)

TPUs use XLA for compilation. You compile your model graph ahead-of-time, similar to Trainium/Inferentia:

```go
package tpu

/*
#cgo CFLAGS: -I/usr/local/lib/python3.9/dist-packages/libtpu/include
#cgo LDFLAGS: -L/usr/local/lib/python3.9/dist-packages/libtpu -ltpu
#include <tpu_driver.h>

// Initialize TPU driver
int init_tpu() {
    return tpu_driver_initialize();
}

// Allocate tensor on TPU HBM
void* alloc_tpu_tensor(size_t size) {
    return tpu_driver_alloc_tensor(size);
}
*/
import "C"
import "unsafe"

// TPUDevice represents a Google TPU chip
type TPUDevice struct {
    deviceID    int
    driver      unsafe.Pointer
    mxuCores    int  // Number of MXU cores (e.g., 2 on v4)
}

// NewTPUDevice initializes a TPU device
func NewTPUDevice(deviceID int) (*TPUDevice, error) {
    if C.init_tpu() != 0 {
        return nil, fmt.Errorf("failed to initialize TPU driver")
    }

    return &TPUDevice{
        deviceID: deviceID,
        mxuCores: 2,  // TPU v4 has 2 MXU cores per chip
    }, nil
}

// MatMul on TPU using MXU (systolic array)
func (d *TPUDevice) MatMul(A, B *Tensor) *Tensor {
    M, K := A.Shape()[0], A.Shape()[1]
    N := B.Shape()[1]

    // Allocate TPU HBM
    d_A := C.alloc_tpu_tensor(C.size_t(M * K * 4))
    d_B := C.alloc_tpu_tensor(C.size_t(K * N * 4))
    d_C := C.alloc_tpu_tensor(C.size_t(M * N * 4))

    // Copy to TPU HBM
    C.tpu_driver_memcpy_h2d(d_A, unsafe.Pointer(&A.Data[0]), C.size_t(M*K*4))
    C.tpu_driver_memcpy_h2d(d_B, unsafe.Pointer(&B.Data[0]), C.size_t(K*N*4))

    // Execute on MXU (128×128 systolic array)
    // Note: In practice, use TensorFlow/JAX to compile XLA HLO
    C.tpu_driver_execute_matmul(d.driver, d_A, d_B, d_C,
        C.int(M), C.int(N), C.int(K))

    // Copy result back
    C_data := make([]float32, M*N)
    C.tpu_driver_memcpy_d2h(unsafe.Pointer(&C_data[0]), d_C, C.size_t(M*N*4))

    return &Tensor{Data: C_data, shape: []int{M, N}}
}
```

### Compiling Models for TPU (JAX Example)

TPUs work best with JAX or TensorFlow. Models are compiled to XLA HLO (High-Level Optimizer):

```bash
# Install JAX for TPU
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Compile model for TPU
python compile_for_tpu.py
```

**compile_for_tpu.py:**
```python
import jax
import jax.numpy as jnp
from jax import jit

# Define transformer forward pass
@jit  # JIT compile to XLA
def transformer_forward(params, input_ids):
    # Embedding
    x = params['embedding'][input_ids]

    # Transformer layers
    for layer in range(12):
        # Multi-head attention (uses TPU MXU)
        x = multi_head_attention(x, params[f'layer_{layer}'])

        # Feed-forward (uses TPU MXU)
        x = feed_forward(x, params[f'ffn_{layer}'])

    # Output projection
    logits = jnp.dot(x, params['output_proj'])
    return logits

# Compile for TPU (creates XLA HLO graph)
compiled_fn = jax.xla_computation(transformer_forward)(params, input_ids)

# Save compiled model
with open('transformer_tpu.hlo', 'wb') as f:
    f.write(compiled_fn.as_serialized_hlo_module_proto())
```

### TPU Pods: Distributed Training

TPUs excel at large-scale distributed training with TPU Pods (interconnected chips):

```go
// TPU Pod configuration
type TPUPod struct {
    chips      []*TPUDevice
    numChips   int
    topology   string  // e.g., "4x4" (16 chips)
}

func NewTPUPod(topology string) *TPUPod {
    // Parse topology (e.g., "4x4" = 16 chips)
    dims := parseTopology(topology)
    numChips := dims[0] * dims[1]

    pod := &TPUPod{
        chips:    make([]*TPUDevice, numChips),
        numChips: numChips,
        topology: topology,
    }

    // Initialize all TPU chips
    for i := 0; i < numChips; i++ {
        pod.chips[i], _ = NewTPUDevice(i)
    }

    // Initialize ICI for all-reduce
    C.tpu_driver_init_collective(C.int(numChips))

    return pod
}

// Data parallel training on TPU Pod
func (p *TPUPod) TrainStep(batches []*Batch) {
    // Distribute batches across TPU chips
    for i, chip := range p.chips {
        go func(chipID int, batch *Batch) {
            // Forward + backward on this TPU
            logits := chip.Forward(batch.Input)
            loss := computeLoss(logits, batch.Target)
            grads := chip.Backward(loss)

            // All-reduce gradients via ICI
            C.tpu_driver_all_reduce(grads.devicePtr, C.int(chipID))

            // Update weights
            chip.UpdateWeights(grads)
        }(i, batches[i])
    }
}
```

### Performance Characteristics

**TPU v5p (highest performance):**
- BF16 compute: 459 TFLOPS (comparable to H100)
- Memory bandwidth: 2400 GB/s
- ICI bandwidth: 4.8 Tbps per chip
- Sweet spot: Large-scale transformer training (>10B params)

**TPU v5e (cost-optimized):**
- BF16 compute: 197 TFLOPS (comparable to A100)
- Memory bandwidth: 820 GB/s
- Cost: 55% cheaper than v5p
- Sweet spot: Medium models, inference

**Comparison:**

| Workload | GPU (H100) | TPU v5p | TPU v5e | AWS Trainium |
|----------|-----------|---------|---------|--------------|
| Training (GPT-2, 1.5B params) | 6 hr | 7 hr | 10 hr | 10 hr |
| Training cost (GCP/AWS) | $200 | $180 | $100 | $80 |
| Inference (ms/token) | 3 | 4 | 6 | 8 |
| Throughput (tokens/sec) | 50K | 45K | 30K | 25K |

### When to Use TPUs

**Use TPUs when:**
- ✅ Training very large models (>10B params) in Google Cloud
- ✅ Using TensorFlow or JAX (seamless integration)
- ✅ Need to scale to TPU Pods (thousands of chips)
- ❌ Avoid for: PyTorch (limited support), non-Google Cloud, small models

**Use GPUs when:**
- ✅ Need PyTorch (best GPU support)
- ✅ Rapid prototyping (no XLA compilation)
- ✅ On-premises or multi-cloud
- ✅ Flexible workloads (not just transformers)

### Practical GCP Setup

```bash
# Create TPU v5e instance (cost-optimized)
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central1-a \
  --accelerator-type=v5litepod-8 \
  --version=tpu-vm-tf-2.14.0

# SSH to TPU VM
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central1-a

# Install JAX
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Run training
python train_transformer.py --tpu_name=my-tpu
```

### Cost Analysis (GCP us-central1, Jan 2025)

| Instance | Hourly Cost | Use Case | Cost per 1B token training |
|----------|-------------|----------|---------------------------|
| a3-highgpu-8g (8× H100) | $31.22 | GPU training | $62 |
| v5p-8 (TPU v5p Pod) | $27.35 | TPU training (highest perf) | $55 |
| v5e-8 (TPU v5e Pod) | $12.00 | TPU training (cost-opt) | $30 |

---

## 8. Azure AI Accelerators (Maia & Cobalt)

Microsoft's custom silicon for AI workloads in Azure, announced in late 2023. **Maia** is an AI accelerator (similar to TPU/Trainium), and **Cobalt** is an Arm-based CPU (similar to Graviton).

### Overview

| Chip | Purpose | Status | Key Features |
|------|---------|--------|--------------|
| **Maia 100** | AI Training/Inference | Preview (2024) | 105B transistors, optimized for GPT/Llama |
| **Cobalt 100** | General Compute | Preview (2024) | Arm Neoverse V2, 128 cores |

**Why Maia/Cobalt?**
- Integration: Deep Azure ecosystem (OpenAI, Bing, Office)
- Cost: Target 50% cost reduction vs NVIDIA GPUs
- Power efficiency: Custom silicon optimized for Microsoft's workloads
- Availability: Exclusive to Azure (not publicly available yet)

### Maia 100 Architecture

**Maia 100 (AI Accelerator):**
- 105 billion transistors (5nm process)
- Optimized for transformer workloads (GPT-4, Llama)
- Custom interconnect for multi-chip scaling
- Tight integration with Azure OpenAI Service

**Current Status:**
- Limited preview (2024-2025)
- Primarily used internally by Microsoft (OpenAI models)
- Public API access planned but not yet available

### Azure Alternatives (Available Now)

Since Maia is not yet publicly available, Azure offers:

**NCads H100 v5 (NVIDIA H100):**
- Standard GPU instances with H100 GPUs
- Full CUDA/cuDNN support
- Available now in select regions

**ND A100 v4 (NVIDIA A100):**
- Mature, widely available GPU instances
- 8× A100 GPUs per VM
- InfiniBand networking for multi-node

```bash
# Create Azure VM with H100 GPUs
az vm create \
  --name my-gpu-vm \
  --resource-group my-rg \
  --image microsoft-dsvm:ubuntu-hpc:2004:latest \
  --size Standard_ND96asr_v4 \  # 8× A100
  --generate-ssh-keys

# SSH and install drivers
ssh azureuser@<vm-ip>
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
```

### Integration Pattern (Hypothetical Maia SDK)

When Maia becomes available, integration will likely follow Azure ML SDK patterns:

```go
// Hypothetical Maia integration (not yet available)
package maia

import "github.com/Azure/azure-sdk-for-go/sdk/ai/azmaia"

type MaiaDevice struct {
    client *azmaia.Client
    deviceID string
}

func NewMaiaDevice(subscriptionID, deviceID string) (*MaiaDevice, error) {
    client, err := azmaia.NewClient(subscriptionID, nil)
    if err != nil {
        return nil, err
    }

    return &MaiaDevice{
        client:   client,
        deviceID: deviceID,
    }, nil
}

// Forward pass on Maia (via Azure ML)
func (d *MaiaDevice) Forward(input []int) (*Tensor, error) {
    // Submit inference job to Maia cluster
    ctx := context.Background()
    result, err := d.client.RunInference(ctx, azmaia.InferenceRequest{
        ModelID: "my-transformer",
        Input:   input,
        Device:  d.deviceID,
    })

    if err != nil {
        return nil, err
    }

    return &Tensor{
        Data:  result.Logits,
        shape: []int{len(input), vocabSize},
    }, nil
}
```

### Azure Cost Comparison (East US, Jan 2025)

| Instance | Hourly Cost | GPUs/Accelerators | Use Case |
|----------|-------------|-------------------|----------|
| NC A100 v4 (8× A100) | $27.20 | 8× A100 40GB | GPU training |
| NCads H100 v5 (8× H100) | $38.88 | 8× H100 80GB | Highest GPU performance |
| Maia 100 (when available) | TBD (~$15-20?) | Maia chips | Cost-optimized AI training |

### When to Use Azure AI Accelerators

**Use Maia (when available) when:**
- ✅ Already using Azure ecosystem (Azure ML, OpenAI Service)
- ✅ Training transformers (GPT, Llama, T5)
- ✅ Cost optimization is critical
- ❌ Avoid for: Non-Azure deployments, need CUDA flexibility

**Use Azure NVIDIA GPUs (now) when:**
- ✅ Need GPUs immediately (Maia not available)
- ✅ Standard CUDA workflows
- ✅ Multi-cloud portability (can move to AWS/GCP)

---

## 9. Complete Accelerator Comparison

| Accelerator | Best For | Cost (Training) | Cost (Inference) | Availability | Ecosystem |
|-------------|----------|-----------------|------------------|--------------|-----------|
| **NVIDIA GPU** | Everything | $$$ | $$ | Excellent | PyTorch/TF/JAX |
| **Apple ANE** | Mac inference | N/A | $ (on-device) | M-series Macs | Core ML |
| **AWS Trainium** | AWS training | $$ | N/A | Good (AWS) | PyTorch/TF |
| **AWS Inferentia2** | AWS inference | N/A | $ | Good (AWS) | PyTorch/TF |
| **Google TPU** | GCP large-scale | $$ | $$ | Good (GCP) | JAX/TF |
| **Azure Maia** | Azure (future) | $? | $? | Preview | TBD |
| **AMD GPU** | Alternative to NVIDIA | $$ | $$ | Growing | ROCm |

### Decision Matrix

**Choose NVIDIA GPUs if:**
- You need maximum flexibility (PyTorch, CUDA)
- You're prototyping or debugging
- You need lowest latency (<3ms inference)
- You're deploying on-premises or multi-cloud

**Choose Cloud Custom Accelerators (Trainium/TPU/Maia) if:**
- You're committed to one cloud (AWS/GCP/Azure)
- Cost is a major concern (30-50% savings)
- You're training large models (>1B params)
- You can use their frameworks (TensorFlow, JAX, Neuron SDK)

**Choose CPU (with SIMD/SVE) if:**
- Model is small (<100M params)
- Inference latency <100ms is acceptable
- You want maximum portability
- No GPU/accelerator available

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
