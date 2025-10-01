# NVIDIA Jetson Orin NX Performance Analysis

## Hardware Specifications

**CPU**: ARM Cortex-A78AE (8 cores)
- Part: 0xd42
- Features: NEON (no SVE/SVE2)
- Clock: ~2 GHz

**GPU**: Integrated Orin
- Compute Capability: 8.7 (Ampere architecture)
- SMs: 8 (vs 40 on T4)
- Memory: 15.29 GB unified
- Clock: 0.92 GHz

## Performance Results (1024×1024 FP64)

| Backend | GFLOPS | vs Naive | Notes |
|---------|--------|----------|-------|
| **Naive CPU** | 1.25 | 1× | Single-threaded, no optimization |
| **NEON** | 0.93 | 0.74× | **SLOWER than naive!** ⚠️ |
| **OpenBLAS** | 92 | 73× | **Clear winner** ✅ |
| **CUDA (Orin)** | 23 | 18× | Good but not beating OpenBLAS |

## Key Findings

### 1. NEON is Slower Than Naive (!)

**Problem**: NEON implementation (0.93 GFLOPS) is 25% slower than naive (1.25 GFLOPS).

**Root Causes**:

1. **Inefficient memory access pattern**:
   ```c
   // Current NEON code (matmul_neon.c line 44-46):
   for (int64_t vl_idx = 0; vl_idx < NEON_WIDTH; vl_idx++) {
       b_vals[0] = b[(l + 0) * n + j];  // Strided access
       b_vals[1] = b[(l + 1) * n + j];  // Non-contiguous
   }
   ```
   Matrix B is accessed with stride `n`, causing cache misses.

2. **No cache blocking**: The NEON code doesn't use tiling, so it thrashes the cache on larger matrices.

3. **Overhead of NEON intrinsics**: For small vectors (2× float64), the NEON overhead may exceed benefits.

**Fixes Needed**:
- Add cache blocking/tiling (like OpenBLAS does)
- Transpose B or use better memory layout
- Consider 4-way unrolling to amortize overhead
- Benchmark against pure cache-blocked naive code

### 2. OpenBLAS Dominates Everything

**Performance**: 92 GFLOPS - absolutely crushing all other options.

**Why it wins**:
- Highly optimized kernels with cache blocking
- Assembly-level tuning for Cortex-A78
- Probably uses 4×4 or 8×8 register blocking
- Prefetching and software pipelining
- Years of optimization work

**Comparison**:
- 4× faster than best GPU result (23 GFLOPS)
- 73× faster than naive
- 99× faster than NEON (!)

**Lesson**: On CPU-bound workloads with good BLAS libraries, the GPU may not win, especially for FP64.

### 3. GPU Performance: Good But Not Great

**CUDA Scaling (FP64)**:
| Size | GFLOPS | % of Peak (0.25 TFLOPS) |
|------|--------|-------------------------|
| 64 | 1.47 | 0.6% |
| 128 | 4.77 | 1.9% |
| 256 | 12.6 | 5.0% |
| 512 | 20.8 | 8.3% |
| 1024 | 22.4 | 9.0% |
| 2048 | 23.6 | 9.4% |
| 4096 | 24.1 | 9.7% |

**Observations**:

1. **Efficiency plateaus at ~10%** of theoretical FP64 peak
   - Orin theoretical: 0.25 TFLOPS FP64
   - Achieved: ~24 GFLOPS = 9.6% efficiency

2. **GPU is overhead-bound below 512×512**
   - Data transfer costs dominate for small matrices
   - Break-even point is around 256×256

3. **Why GPU loses to OpenBLAS**:
   - **FP64 is slow on GPUs**: Orin has 1:64 FP64:FP32 ratio (like most consumer GPUs)
   - **Unified memory overhead**: Data copies between CPU/GPU
   - **Small problem size**: 1024×1024 isn't large enough to saturate GPU
   - **OpenBLAS is just really good**: 92 GFLOPS from 8 CPU cores is exceptional

4. **When GPU would win**:
   - **FP32/FP16**: GPU has 64× more throughput
   - **Larger matrices**: 4096×4096 or bigger
   - **Batch operations**: Multiple small matrices
   - **Training workloads**: Backprop benefits from GPU

## Comparison with AWS g5g (T4 GPU)

Expected results on g5g:

**Graviton2 CPU** (Neoverse N1, 32 cores):
- Naive: ~0.15 GFLOPS (single-thread)
- NEON: ~0.3 GFLOPS (2× speedup if optimized)
- OpenBLAS: ~20-40 GFLOPS (32 cores)

**T4 GPU**:
- 40 SMs (vs 8 on Orin) = 5× more compute
- Higher clocks
- Expected: 100-150 GFLOPS FP64 @ 1024×1024
- Expected: 200-300 GFLOPS FP64 @ 4096×4096

**Verdict**: T4 should beat Graviton2 + OpenBLAS for FP64 at scale, but Orin's OpenBLAS beats Orin GPU.

## ARM Performance Libraries (Arm Compute Library)

**What is it**:
- ARM's official optimized library
- Includes BLAS, convolution, pooling, etc.
- Hand-tuned for Neoverse (Graviton) and Cortex-A (Orin)
- Free and open-source

**Expected performance**:
- **Better than OpenBLAS on ARM**: 10-30% faster
- **Uses NEON + cache blocking**: Properly optimized
- **May use SVE on Graviton3/4**: Scalable vectors
- **Expected on Orin**: 100-120 GFLOPS (vs 92 from OpenBLAS)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install libarmcl-dev

# Or build from source
git clone https://github.com/ARM-software/ComputeLibrary.git
cd ComputeLibrary
scons Werror=0 debug=0 neon=1 opencl=0 embed_kernels=1 \
      os=linux arch=armv8-a
```

**Integration**:
- Similar to OpenBLAS: CBLAS-compatible interface
- May need custom bindings for GEMM

## Recommendations

### For Teaching/Learning:
1. **Fix NEON implementation**:
   - Add cache blocking (64×64 tiles)
   - Improve memory access patterns
   - This is valuable for understanding optimization

2. **Add ARM Compute Library backend**:
   - Show best-in-class ARM performance
   - Demonstrate proper NEON usage
   - Compare against hand-rolled NEON

### For Production:
1. **CPU workloads**: Use OpenBLAS or ARM Compute Library
2. **GPU workloads**:
   - Use FP32/FP16, not FP64
   - Ensure matrices are large enough (>2048)
   - Consider batching small operations
3. **Hybrid**: Offload FP32 to GPU, keep FP64 on CPU

### Next Steps:
1. ✅ Identify NEON performance issue
2. ⏳ Implement ARM Compute Library backend
3. ⏳ Test FP32 GPU performance (should be ~64× faster)
4. ⏳ Test on real g5g with T4 GPU
5. ⏳ Test on Graviton3/4 with SVE
6. ⏳ Optimize NEON with cache blocking

## Educational Value

This demonstrates several key lessons:

1. **SIMD isn't magic**: Bad SIMD code is slower than good scalar code
2. **Memory matters more than compute**: Cache blocking > vectorization
3. **Use libraries**: OpenBLAS represents decades of optimization work
4. **GPU isn't always faster**: Especially for FP64, small problems, or when CPU library is excellent
5. **Know your hardware**: FP64 vs FP32 ratio matters enormously
6. **Measure, don't assume**: Benchmarking reveals surprising results

## References

- [ARM Compute Library](https://github.com/ARM-software/ComputeLibrary)
- [ARM Performance Libraries](https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Libraries)
- [NVIDIA Jetson Orin Specs](https://developer.nvidia.com/embedded/jetson-orin)
- [OpenBLAS Documentation](https://www.openblas.net/)
- [GPU GEMM Optimization Guide](https://docs.nvidia.com/cuda/cublas/)
