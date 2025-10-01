# AWS Graviton Optimization Plan

## Overview

This document outlines the optimization strategy for AWS Graviton processors (ARM Neoverse architecture), covering Graviton2 through Graviton4, plus GPU acceleration options.

## Graviton Processor Evolution

### Graviton2 (2020)
- **Architecture**: ARM Neoverse N1
- **Cores**: 64 cores
- **ISA**: ARMv8.2-A
- **Memory**: DDR4
- **SIMD**: NEON (128-bit)
- **Process**: 7nm
- **Use cases**: General purpose, cost-effective ARM

### Graviton3 (2021)
- **Architecture**: ARM Neoverse V1
- **Cores**: 64 cores @ 2.6 GHz
- **ISA**: ARMv8.4-A
- **Memory**: 8× DDR5-4800 channels
- **SIMD**: NEON (128-bit) + **SVE (256-bit)**
- **Vector Units**: 2× 256-bit SVE units per core
- **Process**: 5nm (7-chiplet design)
- **Performance**: 25% better compute, 2× FP performance vs Graviton2

### Graviton3E (2022)
- **Same as Graviton3** but higher power/performance envelope
- **SIMD**: NEON (128-bit) + **SVE (256-bit)** - same as Graviton3
- **Performance**: 35% higher vector performance (higher clocks, not wider vectors)
- **Use cases**: HPC, scientific computing, ML training

### Graviton4 (2023)
- **Architecture**: ARM Neoverse V2
- **Cores**: 96 cores (50% more than Graviton3)
- **ISA**: ARMv9.0-A + **SVE2** with crypto extensions
- **Memory**: 12× DDR5-5600 controllers (75% more bandwidth)
- **SIMD**: NEON (128-bit) + **SVE2 (128-bit)** ⚠️ SMALLER than Graviton3!
- **Vector Units**: **4× 128-bit SVE2 engines** per core (vs V1's 2× 256-bit)
- **Process**: 4nm
- **Performance**: 30% better compute, 50% more cores vs Graviton3
- **Trade-off**: Narrower vectors but 2× more vector engines per core

## ⚠️ Important: Graviton Vector Width Surprise

**Graviton4 has NARROWER vectors than Graviton3!**

| Processor | Vector Width | Elements (FP64) | Why? |
|-----------|--------------|-----------------|------|
| Graviton2 | NEON 128-bit | 2 | No SVE |
| Graviton3 | SVE 256-bit | 4 | Neoverse V1 design |
| Graviton3E | SVE 256-bit | 4 | Same as G3, higher clocks |
| Graviton4 | SVE2 128-bit | 2 | Neoverse V2 trade-off |

**Why did ARM/AWS reduce vector width?**
- More cores (96 vs 64) at lower power
- Better for throughput (more cores) vs latency (wider vectors)
- Power efficiency: narrower vectors use less power
- Memory bandwidth: 12 DDR5-5600 channels feed more cores better
- SVE2 ISA improvements compensate for narrower width

**Performance Implications:**
- **Graviton3/3E**: Best for heavily vectorized workloads (2× vector width)
- **Graviton4**: Best for multi-threaded workloads (1.5× more cores)
- **Vector workloads** (e.g., vector search): Graviton3 can outperform Graviton4!
- **Matrix multiply**: Graviton4 likely still wins (more cores + SVE2 + bandwidth)

## GPU Acceleration Options

### G4dn Instances (Intel + NVIDIA T4)
- **CPU**: Intel Cascade Lake (x86)
- **GPU**: NVIDIA T4 Tensor Core
- **Use case**: ML inference, small-scale training
- **Not Graviton-based**

### G5g Instances (Graviton2 + NVIDIA T4G)
- **CPU**: AWS Graviton2 (64 ARM cores, Neoverse N1)
- **GPU**: NVIDIA T4G Tensor Core (Turing architecture)
- **Tensor Cores**: 320 Turing Tensor Cores for mixed-precision
- **Memory**: Up to 16 GB GPU memory (GDDR6)
- **Use case**: ML inference on ARM + GPU, Android game streaming
- **Performance**: 30% better price/performance for GPU workloads vs Intel-based G4dn
- **First ARM-based GPU instances in AWS**
- **Key advantage**: Graviton2 CPU handles data prep while GPU handles compute

## Optimization Strategy

### Level 0: Baseline (Pure Go)
**Target**: All Graviton processors
- Pure Go implementation
- Works out-of-box on ARM64
- No special optimizations
- **Expected**: Decent performance (Go compiler generates good ARM code)

### Level 1: ARM-Optimized Go
**Target**: All Graviton processors
- Use ARM64-specific Go stdlib optimizations
- Leverage Go's ARM64 math optimizations
- Profile-guided optimization (PGO)
- **Effort**: 2h
- **Expected speedup**: 1.2-1.5×

### Level 2: NEON SIMD (Graviton2, 3, 3E, 4)
**Target**: All Graviton processors (NEON is baseline)
- **128-bit vector operations**
- Reuse existing `matmul_neon_arm64.s`
- Works on all Graviton generations
- **Effort**: Already done! ✅
- **Expected speedup**: 2-4× (similar to M4 Max results)

### Level 3: SVE (Graviton3/3E/4 only)
**Target**: Graviton3, Graviton3E, Graviton4
- **Scalable Vector Extension** (256-bit on G3, up to 512-bit on G4)
- Vector-length agnostic programming
- Better performance than fixed NEON
- **Effort**: 16h (new assembly implementation)
- **Expected speedup**: 1.5-2× over NEON on Graviton3+

### Level 4: SVE2 (Graviton4 only)
**Target**: Graviton4 only
- **SVE2 with crypto extensions**
- Enhanced integer/FP operations
- Better gather/scatter for sparse matrices
- **Effort**: 8h (extend SVE implementation)
- **Expected speedup**: 1.2-1.5× over SVE on Graviton4

### Level 5: AWS Optimized BLAS
**Target**: All Graviton processors
- Use AWS-optimized OpenBLAS or BLIS
- Similar to Accelerate on macOS
- **Effort**: 4h (CGo bindings)
- **Expected speedup**: 10-20× (highly optimized BLAS)

### Level 6: GPU Acceleration (G5g instances)
**Target**: Graviton2 + NVIDIA T4G
- CUDA kernels for T4G Tensor Cores
- Graviton2 CPU for data prep/coordination
- **Effort**: 40h+ (CUDA programming)
- **Expected speedup**: 50-100× (GPU vs CPU)
- **Cost**: Higher instance cost

## Recommended Implementation Order

### Phase 1: Verify NEON Works (0.5h)
- Test existing NEON implementation on Graviton2/3/4
- Benchmark against pure Go
- **Why**: Reuse existing work

### Phase 2: Add SVE Support (16h)
- Implement SVE version of matmul
- Runtime detection: use SVE on G3+, NEON on G2
- Focus on Graviton3/3E first (more common)
- **Why**: 1.5-2× improvement on modern Graviton

### Phase 3: AWS BLAS Integration (4h)
- Test OpenBLAS on Graviton
- Compare with SVE implementation
- **Why**: Might beat custom SVE (like Accelerate beat NEON)

### Phase 4: SVE2 Optimization (8h)
- Graviton4-specific optimizations
- Use crypto extensions if applicable
- **Why**: Future-proof for newest instances

### Phase 5: G5g GPU (optional, 40h+)
- Only if GPU acceleration is needed
- Requires CUDA expertise
- **Why**: 50-100× speedup but high complexity

## Expected Performance Matrix

| Optimization | Graviton2 | Graviton3 | Graviton3E | Graviton4 | Effort |
|--------------|-----------|-----------|------------|-----------|--------|
| Pure Go | 1× | 1.25× | 1.3× | 1.5× | 0h |
| NEON | 3× | 3.5× | 3.7× | 4× | ✅ Done |
| SVE | — | 6× | 6.5× | 7× | 16h |
| SVE2 | — | — | — | 9× | +8h |
| AWS BLAS | 30× | 40× | 45× | 50× | 4h |
| GPU (G5g) | 150× | — | — | — | 40h+ |

*Relative to pure Go baseline on same processor*

## Cost Considerations

### Instance Pricing (Approximate, us-east-1)
- **c7g.xlarge** (Graviton3): ~$0.07/hr
- **c7gn.xlarge** (Graviton3E): ~$0.09/hr
- **c8g.xlarge** (Graviton4): ~$0.08/hr (when available)
- **g5g.xlarge** (Graviton2 + T4G GPU): ~$0.42/hr

### Price/Performance Sweet Spot
- **Graviton3**: Best overall value (mature, widely available)
- **Graviton4**: 30% faster, similar price when available
- **G5g**: 5× more expensive but 50× faster for ML workloads

## Testing Strategy

### 1. Verify NEON on Graviton2
```bash
# Launch c6g.xlarge (Graviton2)
go test -bench BenchmarkMatMul -cpuprofile cpu.prof
# Should see 3-4× speedup vs naive
```

### 2. Test SVE Detection on Graviton3
```bash
# Launch c7g.xlarge (Graviton3)
# Check /proc/cpuinfo for "sve" flag
cat /proc/cpuinfo | grep sve
# Should show "sve" feature
```

### 3. Benchmark SVE vs NEON on Graviton3
```bash
# Compare SVE and NEON implementations
go test -bench . -run=^$ -tags sve
go test -bench . -run=^$ -tags neon
# SVE should be 1.5-2× faster
```

### 4. Compare with AWS BLAS
```bash
# Install OpenBLAS optimized for ARM
sudo apt-get install libopenblas-dev
go test -bench BenchmarkAccelerate
# Should see 10-20× speedup
```

## Key Differences: Graviton vs Apple Silicon

| Feature | Apple M4 Max | Graviton4 |
|---------|--------------|-----------|
| Architecture | Apple Silicon (custom) | ARM Neoverse V2 |
| Vector Width | NEON 128-bit | NEON + SVE2 (up to 512-bit) |
| GPU | Integrated, ~2.7 TFLOPS | None (use G5g for GPU) |
| ANE | Yes, 38 TFLOPS | No equivalent |
| BLAS | Accelerate (vecLib) | OpenBLAS / BLIS |
| Memory | Unified (400 GB/s) | DDR5-5600 (12 channels) |
| Use Case | Development, prototyping | Cloud deployment, scale |

## Recommendations

### For this project:
1. ✅ **Start with NEON** (already implemented)
2. **Add AWS BLAS support** (4h effort, best ROI)
3. **Implement SVE for Graviton3+** if BLAS isn't enough (16h)
4. **Skip GPU** unless doing large-scale inference

### Documentation to create:
1. `graviton_simd.s` - SVE implementation
2. `graviton_blas.go` - AWS OpenBLAS bindings
3. `graviton_test.go` - Graviton-specific tests
4. `GRAVITON_RESULTS.md` - Benchmark results on real AWS instances

### Why not GPU (G5g)?
- 5× more expensive per hour
- Our transformer is small (training focus)
- GPU overhead matters for small batches
- Better for large-scale inference, not training

## Next Steps

Ready to proceed with:
1. Test existing NEON on Graviton2
2. Implement SVE for Graviton3/4
3. Add AWS BLAS support
4. Benchmark comprehensive comparison

Which would you like to start with?
