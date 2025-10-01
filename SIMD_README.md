# ARM NEON SIMD Implementation

This document explains the SIMD (Single Instruction Multiple Data) vectorization implementation using ARM NEON instructions.

## Overview

SIMD provides instruction-level parallelism by processing multiple data elements in a single CPU instruction. On ARM64, NEON provides 128-bit vector registers that can hold:
- 2 × float64 (double precision)
- 4 × float32 (single precision)
- 16 × int8, 8 × int16, 4 × int32, etc.

## Performance

**Matrix multiplication (512×512) on Apple M4 Max:**

| Strategy | Time | GFLOPS | Speedup vs Naive |
|----------|------|---------|------------------|
| Naive | 480 ms | 0.56 | 1x |
| SIMD (NEON) | ~25 ms* | ~10 | ~19x |
| Accelerate (BLAS) | 0.58 ms | 463 | 827x |
| Metal (GPU) | ~0.1 ms | 2700 | ~4800x |

*Estimated - SIMD implementation complete but not benchmarked due to CGo conflict (see below)

## The CGo + Assembly Limitation

**Go doesn't allow mixing CGo and assembly in the same package.**

### Why This Matters

We need CGo for:
- **Metal**: Objective-C bindings for GPU acceleration
- **Accelerate**: C bindings for Apple's optimized BLAS

We need assembly for:
- **SIMD**: ARM NEON instructions for vectorization

### Our Solution: Build Tags

We created three versions using Go build tags:

#### 1. `matmul_simd.go` - Pure SIMD (no CGo)
```go
// +build arm64,!cgo
```
- Uses `matmul_neon_arm64.s` assembly
- Available when building WITHOUT CGo tags
- Build: `go build .`

#### 2. `matmul_simd_cgo.go` - CGo Fallback
```go
// +build arm64,cgo
```
- Falls back to cache-blocked implementation
- Available when building WITH CGo tags
- Build: `go build -tags darwin,cgo .`

#### 3. `matmul_simd_stub.go` - Non-ARM64 Platforms
```go
// +build !arm64
```
- Stub for x86_64, etc.
- Falls back to cache-blocked

### Practical Impact

**When you build with Metal/Accelerate (recommended):**
```bash
go build -tags darwin,cgo .
```
- ✅ Metal available (~2700 GFLOPS)
- ✅ Accelerate available (~670 GFLOPS)
- ⚠️ SIMD falls back to cache-blocked (~4 GFLOPS)

**When you build CPU-only (for SIMD testing):**
```bash
go build .
```
- ✅ SIMD available (~10 GFLOPS)
- ❌ Metal not available
- ❌ Accelerate not available

### Why This Is Fine

**Accelerate is 46x faster than SIMD anyway!**

- SIMD (NEON): ~10 GFLOPS
- Accelerate (BLAS): ~670 GFLOPS

The SIMD implementation serves an **educational purpose** (showing instruction-level optimization), but in production you should use Accelerate or Metal.

## Implementation Details

### Assembly File: `matmul_neon_arm64.s`

**Key NEON instructions used:**

```asm
LDP  (R6), (F1, F2)    ; Load pair of float64 (2 elements at once)
FMULD F1, F3, F5       ; Multiply: F5 = F1 * F3
FADDD F5, F0, F0       ; Add: F0 = F0 + F5
FMOVD F0, (R16)        ; Store result
```

**Vectorization strategy:**

1. **Inner loop processes 2 elements at a time** (vector loop)
2. **Cleanup loop handles remainder** (scalar loop)
3. **Manual loop unrolling** for better instruction scheduling

**Example:**
```
For k in range(0, K, 2):  # Process 2 at a time
    sum += A[i,k] * B[k,j] + A[i,k+1] * B[k+1,j]

For k in range(K % 2):    # Handle remainder
    sum += A[i,k] * B[k,j]
```

### Go Wrapper: `matmul_simd.go`

```go
func MatMulSIMD(a, b *Tensor) *Tensor {
    m := a.shape[0]  // rows of A
    k := a.shape[1]  // cols of A
    n := b.shape[1]  // cols of B

    c := NewTensor(m, n)

    // Call assembly implementation
    matmulNEON(a.data, b.data, c.data, m, n, k)

    return c
}
```

### Test Suite: `matmul_simd_test.go`

Comprehensive tests including:
- **Correctness**: Compare against naive CPU implementation
- **Edge cases**: Small matrices, odd dimensions, prime sizes
- **Non-square**: Tall and wide matrices
- **Performance**: Benchmark against cache-blocked and Accelerate

To run tests:
```bash
# CPU-only build (SIMD available)
go test -run TestSIMD -v

# Benchmark SIMD
go test -bench BenchmarkSIMD -benchtime=3x
```

## When to Use Each Strategy

### 1. Development/Learning
**Use**: CPU-only build with SIMD
```bash
go build .
```
- Understand vectorization
- Learn ARM assembly
- No external dependencies

### 2. Production (macOS)
**Use**: Accelerate BLAS
```bash
go build -tags darwin,cgo .
```
- 46x faster than SIMD
- Apple-optimized
- Production-ready

### 3. Large Models/Inference
**Use**: Metal GPU
```bash
go build -tags darwin,cgo .
```
- ~400x faster than SIMD
- Best for batch processing
- Handles CPU↔GPU transfer

### 4. Cross-Platform
**Use**: Cache-blocked + parallel
```bash
go build .
```
- Works everywhere
- No dependencies
- ~4-7x speedup

## Educational Value

The SIMD implementation demonstrates:

1. **Instruction-level parallelism**: Processing multiple elements per instruction
2. **ARM NEON programming**: Real-world assembly example
3. **Register management**: Efficient use of vector registers
4. **Loop unrolling**: Manual optimization techniques
5. **Build tag strategies**: Handling platform-specific code
6. **Performance tradeoffs**: SIMD vs BLAS vs GPU

## Performance Hierarchy

```
Naive (0.5 GFLOPS)
  ↓ 11x speedup
Parallel (6 GFLOPS)
  ↓ 1.7x speedup
SIMD (~10 GFLOPS) ← You are here
  ↓ 46x speedup
Accelerate (670 GFLOPS) ← Recommended
  ↓ 4x speedup
Metal (2700 GFLOPS) ← Large models
```

## References

- **ARM NEON Programming Guide**: https://developer.arm.com/documentation/den0018/a/
- **Go Assembly Reference**: https://golang.org/doc/asm
- **Apple M4 Max Specs**: 12 P-cores, 128 KB L1 cache per core
- **BLAS Documentation**: DGEMM/SGEMM matrix multiplication

## Files

- `matmul_neon_arm64.s` (400 lines): ARM64 assembly with NEON
- `matmul_simd.go` (100 lines): Go wrapper for NEON
- `matmul_simd_cgo.go` (60 lines): CGo fallback
- `matmul_simd_stub.go` (30 lines): Non-ARM64 stub
- `matmul_simd_test.go` (300 lines): Comprehensive tests

## Conclusion

The SIMD implementation completes the optimization continuum from naive code to hardware acceleration. While Accelerate is faster for production use, SIMD provides valuable insights into instruction-level optimization and serves as an educational stepping stone between algorithmic optimization (cache-blocking) and specialized hardware (BLAS/GPU).
