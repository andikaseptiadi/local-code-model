# Matrix Multiplication Optimization Journey

## Project Goal

This project is a **teaching resource** that demonstrates the optimization journey from naive matrix multiplication to production-grade performance across different hardware accelerators. The goal is to show **why** certain optimizations work and **when** they fail.

## Key Insight: Performance Is Counterintuitive

Our most important finding: **Naive SIMD vectorization made performance 25% WORSE**.

This teaches a critical lesson that theoretical computer science often misses: **memory access patterns matter more than compute optimization**.

## The Optimization Ladder

### Level 0: Naive (Baseline)
```go
// Triple-nested loop, no optimization
for i := 0; i < m; i++ {
    for j := 0; j < n; j++ {
        for k := 0; k < K; k++ {
            C[i*n+j] += A[i*K+k] * B[k*n+j]
        }
    }
}
```
**Performance**: 1.25 GFLOPS (Jetson Orin NX, 1024×1024)

### Level 1: NEON SIMD (FAILED)
```c
// Process 2 float64 at once with NEON
float64x2_t a_vec = vld1q_f64(&A[i*k+l]);
float64x2_t b_vec = vld1q_f64(b_vals);  // But b_vals are strided!
```
**Performance**: 0.93 GFLOPS ⚠️ (25% SLOWER!)

**Why it failed**:
- Matrix B accessed with stride=1024 (non-contiguous)
- Cache misses destroyed performance
- Extra data marshaling added overhead
- **Lesson**: Vectorization without memory optimization makes things worse

### Level 2: Parallel (SUCCESS)
```go
// Divide work across CPU cores
numWorkers := runtime.NumCPU()
// Each worker processes m/numWorkers rows
```
**Performance**: ~10 GFLOPS (8 cores)
**Speedup**: 8× (linear scaling)

### Level 3: OpenBLAS (HUGE SUCCESS)
```c
// Use production library with:
// - Multi-level cache blocking
// - Register blocking
// - Prefetching
// - Hand-tuned assembly
cblas_dgemm(...)
```
**Performance**: 92 GFLOPS ✅
**Speedup**: 73× vs naive, 99× vs broken NEON!

**Why it won**:
- L1/L2/L3 cache blocking
- 4×4 or 8×8 register blocks
- Architecture-specific tuning
- 10+ years of optimization work

### Level 4: CUDA GPU (MIXED)
```c
// Offload to NVIDIA GPU
cublasDgemm(...)
```
**Performance**: 23 GFLOPS (FP64)
**Speedup**: 18× vs naive, but **4× slower than OpenBLAS!**

**Why it "lost"**:
- FP64 has 1:64 ratio vs FP32 on consumer GPUs
- Problem size (1024) too small to saturate GPU
- Data transfer overhead
- **Would win with**: FP32 (1.5 TFLOPS), larger matrices (4096+), batch operations

## Hardware Tested

### NVIDIA Jetson Orin NX
- **CPU**: ARM Cortex-A78AE (8 cores, 2 GHz)
- **GPU**: Integrated Orin (8 SMs, compute 8.7, 15 GB unified memory)
- **Features**: NEON (no SVE)

**Results (1024×1024 FP64)**:
- Naive: 1.25 GFLOPS
- NEON: 0.93 GFLOPS (slower!)
- OpenBLAS: 92 GFLOPS (winner!)
- CUDA: 23 GFLOPS

### AWS Graviton (Planned Testing)

**Graviton2** (c6g):
- Neoverse N1, NEON only
- Expected: OpenBLAS ~30-40 GFLOPS (32 cores)

**Graviton3** (c7g):
- Neoverse V1, 2× 256-bit SVE engines
- Expected: OpenBLAS ~60-80 GFLOPS with SVE

**Graviton4** (c8g):
- Neoverse V2, 4× 128-bit SVE2 engines
- Expected: OpenBLAS ~80-100 GFLOPS with SVE2

**AWS g5g** (Graviton2 + T4 GPU):
- T4 has 40 SMs (5× more than Orin)
- Expected CPU: 30-40 GFLOPS (OpenBLAS)
- Expected GPU: 150-200 GFLOPS FP64 @ 1024×1024

## Backend Implementations

### Current Status

| Backend | Status | Platform | Performance Notes |
|---------|--------|----------|-------------------|
| **Naive** | ✅ Complete | All | Baseline reference |
| **Parallel** | ✅ Complete | All | Linear scaling |
| **NEON** | ⚠️ Needs fixing | ARM64 | Currently slower than naive! |
| **SVE/SVE2** | ✅ Complete | Graviton3/4 | Untested on real hardware |
| **OpenBLAS** | ✅ Complete | Linux | 73× speedup, clear winner |
| **CUDA** | ✅ Complete | NVIDIA GPUs | 18× speedup (FP64) |
| **ARM CL** | ⏳ Partial | ARM64 | Needs testing |
| **Apple Metal** | ✅ Complete | macOS | GPU acceleration |
| **Apple ANE** | ✅ Complete | macOS | Neural Engine |

### File Organization

```
Backend Implementations:
- matmul_neon.c, matmul_neon_linux.go    # NEON (needs optimization)
- matmul_sve.c, matmul_sve_linux.go      # SVE/SVE2 for Graviton3/4
- openblas_linux.go, openblas_stub.go    # OpenBLAS (best CPU)
- gpu_cuda_linux.go, gpu_cuda_stub.go    # CUDA (NVIDIA GPUs)
- armcl_linux.go, armcl_wrapper.cpp      # ARM Compute Library (new)

Test Files:
- openblas_test.go                       # OpenBLAS + NEON tests
- gpu_cuda_test.go                       # CUDA tests
- sve_test.go                            # SVE-specific tests
- graviton_test.go                       # Graviton CPU features

Documentation:
- PERFORMANCE_LESSONS.md                 # Key insights
- ORIN_NX_ANALYSIS.md                   # Jetson Orin results
- IMPLEMENTATION_STATUS.md               # Implementation tracking
- G5G_TESTING.md                        # AWS g5g guide
- BUILD_TAGS.md                         # Build system explanation
```

## Key Lessons Learned

### 1. Memory Access > Vectorization

**Wrong approach**: "Add SIMD, get 2× speedup"

**Right approach**:
1. Optimize memory access patterns first
2. Add cache blocking
3. Choose correct data layout
4. Then add SIMD if it helps

**Evidence**: Our NEON code was 25% slower because it had bad memory access.

### 2. Libraries Represent Decades of Work

OpenBLAS achieved 92 GFLOPS through:
- Multi-level cache blocking (L1/L2/L3)
- Register blocking (4×4 or 8×8)
- Prefetching
- Hand-written assembly
- Architecture-specific tuning

**We can't compete with this** - nor should we try. Use libraries in production.

### 3. GPU Isn't Always Faster

GPU lost to CPU + OpenBLAS because:
- FP64 is 64× slower than FP32 on consumer GPUs
- 1024×1024 is too small to saturate GPU
- Data transfer overhead
- OpenBLAS is exceptionally well-optimized

**GPU wins when**:
- Using FP32/FP16 (64× faster)
- Large matrices (4096+)
- Batch operations
- Training workloads

### 4. The Optimization Hierarchy

Priority order (most to least important):
1. **Algorithm** (O(n³) → O(n^2.8)): 10× possible
2. **Memory patterns** (cache blocking): 40× possible
3. **Data layout** (SoA vs AoS): 2-5× possible
4. **Parallelization** (multi-core): 6-8× possible
5. **Vectorization** (SIMD): 2-4× possible (if done right!)

Our NEON code did #5 without #2 and #3, so it failed.

### 5. Always Profile, Never Assume

Assumptions we had:
- ✗ "SIMD is always faster" → Was 25% slower
- ✗ "GPU beats CPU" → OpenBLAS won
- ✗ "More vector width = faster" → Memory bandwidth limited
- ✓ "Libraries are well-optimized" → OpenBLAS proved this

**Lesson**: Measure everything, assume nothing.

## How to Use This Project

### For Learning

1. **Start with naive**: Understand the algorithm
2. **Study PERFORMANCE_LESSONS.md**: See why naive SIMD failed
3. **Compare implementations**: Naive → Parallel → OpenBLAS → GPU
4. **Profile your changes**: Use `perf stat`, `cachegrind`, etc.
5. **Understand the hardware**: Read CPU/GPU architecture docs

### For Teaching

This project demonstrates:
- Why memory hierarchy matters
- The cost of bad memory access
- When SIMD helps vs hurts
- How to use production libraries
- GPU vs CPU tradeoffs
- The importance of profiling

Use it to teach:
- Systems programming
- Performance optimization
- Computer architecture
- Parallel computing
- GPU programming

### For Production

**Don't use this code in production!**

Use instead:
- CPU: OpenBLAS, ARM Compute Library, Intel MKL
- GPU: cuBLAS, ROCm BLAS, oneDNN
- ML: PyTorch, TensorFlow, JAX

This project is for **learning**, not production use.

## Next Steps

### Immediate Fixes Needed

1. **Fix NEON implementation** ⏳:
   - Add 64×64 cache blocking
   - Transpose matrix B or improve access pattern
   - Benchmark against pure cache-blocked code
   - **Goal**: Match or beat naive (1.25 GFLOPS)

2. **Test ARM Compute Library** ⏳:
   - Install on Orin: `apt-get install libarmcl-dev`
   - Complete C++ wrapper integration
   - **Expected**: 100-120 GFLOPS (10-30% better than OpenBLAS)

3. **Test on real Graviton** ⏳:
   - Graviton2 (NEON): Verify baseline
   - Graviton3 (SVE): Test 256-bit vectors
   - Graviton4 (SVE2): Test 4× 128-bit engines
   - Compare with OpenBLAS

4. **Test on g5g** ⏳:
   - T4 GPU should achieve 150-200 GFLOPS (FP64)
   - Compare with Graviton2 + OpenBLAS
   - Test FP32 for 10× improvement

### Future Enhancements

- [ ] Add FP32/FP16 support for GPU
- [ ] Implement Strassen algorithm (O(n^2.8))
- [ ] Add batched operations
- [ ] Mixed precision training
- [ ] Multi-GPU support
- [ ] Distributed matrix multiplication

## Build and Test

### macOS (Apple Silicon)
```bash
go build          # Uses native Metal/ANE backends
go test -v
```

### Linux ARM64 (Graviton/Orin)
```bash
export CGO_ENABLED=1
go build
go test -v -run TestOpenBLAS
go test -v -run TestCUDA    # If NVIDIA GPU
```

### Benchmarking
```bash
# Compare all backends
go test -v -bench BenchmarkG5GComparison -benchtime=3s

# GPU scaling
go test -v -bench BenchmarkCUDAScaling

# SVE on Graviton3/4
go test -v -bench BenchmarkSVEGravitonComparison
```

## Contributing

This is a teaching project. Contributions that improve **educational value** are welcome:

- Better documentation
- More detailed performance analysis
- Additional platform support
- Clearer code comments
- Performance visualizations

**Not welcome**:
- Production-grade optimizations that obscure learning
- Complex code without educational benefit
- Removing "slow" implementations (they teach lessons!)

## References

### Papers & Books
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) - Essential reading
- [Gallery of Processor Cache Effects](https://igoro.com/archive/gallery-of-processor-cache-effects/) - Visual demonstrations
- "Computer Architecture: A Quantitative Approach" - Hennessy & Patterson

### Documentation
- [ARM NEON Optimization Guide](https://developer.arm.com/documentation/102467/latest/)
- [ARM SVE Programming Guide](https://developer.arm.com/documentation/102476/latest/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [OpenBLAS](https://www.openblas.net/)
- [ARM Compute Library](https://github.com/ARM-software/ComputeLibrary)

### Related Projects
- [OpenBLAS](https://github.com/xianyi/OpenBLAS) - Reference CPU BLAS
- [cuBLAS](https://developer.nvidia.com/cublas) - NVIDIA GPU BLAS
- [Eigen](https://eigen.tuxfamily.org/) - C++ template library
- [BLIS](https://github.com/flame/blis) - BLAS-like Library Instantiation Software

## License

This is an educational project. Use it to learn, teach, and understand performance optimization.

For production workloads, use mature libraries like OpenBLAS, cuBLAS, or ARM Compute Library.

---

**Remember**: The goal is not to beat OpenBLAS. The goal is to **understand why** OpenBLAS is so fast and **learn** from our failed attempts. Sometimes the best teacher is a 25% performance regression!
