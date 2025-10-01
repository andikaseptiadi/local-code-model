# Graviton4 Detailed Analysis

**Instance**: c8g.2xlarge (8 vCPUs)
**CPU**: Neoverse V2 (0xd4f)
**Vector**: SVE2 128-bit (4× engines per core)
**Memory**: DDR5

## Performance Across Matrix Sizes

| Size | Single-Thread | Parallel (8 cores) | Speedup | Efficiency |
|------|---------------|--------------------|---------| -----------|
| 128×128 | 0.26 GFLOPS | 1.81 GFLOPS | 6.85× | 85.7% |
| 256×256 | 0.26 GFLOPS | 1.69 GFLOPS | 6.43× | 80.3% |
| 512×512 | 0.26 GFLOPS | 1.86 GFLOPS | 7.20× | 90.0% |
| 1024×1024 | 0.24 GFLOPS | 1.84 GFLOPS | 7.75× | 96.8% |

## Key Observations

### 1. Consistent Single-Thread Performance
- ~0.26 GFLOPS across all sizes (drops to 0.24 at 1024×1024)
- Slight degradation at large sizes suggests cache pressure

### 2. Parallel Efficiency Improves with Size
- **Small (256)**: 80% efficiency - thread coordination overhead
- **Medium (512)**: 90% efficiency - sweet spot
- **Large (1024)**: 97% efficiency - near-perfect scaling

This is **opposite** of what we'd expect if memory bandwidth were the bottleneck!

### 3. Cache Blocking Still Ineffective
- 128: 1.22-1.23× (marginal improvement)
- 256: 1.23× (marginal)
- 512: 1.17× (marginal)
- 1024: 1.31× (first meaningful improvement!)

Cache blocking only helps at 1024×1024, suggesting:
- Smaller matrices fit in L2/L3 cache
- 1024×1024 (~8MB) exceeds cache, blocking helps

## Comparison: All 4 Graviton Generations

### Single-Thread Performance (256×256)

| Generation | GFLOPS | vs G2 | vs G3 | Architecture |
|------------|--------|-------|-------|--------------|
| Graviton2 | 0.15 | Baseline | -32% | Neoverse N1, NEON 128-bit |
| Graviton3 | 0.22 | +47% | Baseline | Neoverse V1, SVE 256-bit |
| Graviton3E | 0.22 | +47% | 0% | Neoverse V1, SVE 256-bit |
| **Graviton4** | **0.26** | **+73%** | **+18%** | **Neoverse V2, SVE2 128-bit** 🏆 |

### Parallel Performance (256×256, 8 cores)

| Generation | GFLOPS | vs G2 | vs G3 | Efficiency |
|------------|--------|-------|-------|------------|
| Graviton2 | 1.18 | Baseline | -33% | 98.2% |
| Graviton3 | 1.75 | +48% | Baseline | 98.6% |
| Graviton3E | 1.74 | +47% | -1% | 96.7% |
| **Graviton4** | **1.87** | **+58%** | **+7%** | **88.9%** |

## The Graviton4 Paradox

**Graviton4 is fastest but least efficient. Why?**

### Hypothesis 1: Memory Bandwidth (DISPROVEN)
If memory bandwidth were the issue, efficiency would **decrease** with larger matrices. But we see the **opposite** - efficiency improves from 80% (256×256) to 97% (1024×1024).

### Hypothesis 2: Thread Coordination Overhead
- Graviton4 has more sophisticated out-of-order execution
- Thread synchronization may have higher latency
- Small matrices don't amortize thread startup cost
- **Evidence**: Efficiency improves dramatically with size

### Hypothesis 3: SVE2 4× Engines Under-Utilized
Graviton4 architecture:
- **4× 128-bit SVE2 engines** per core (vs G3's 2× 256-bit)
- Designed for **higher throughput** with more parallel operations
- Current code is **pure Go**, doesn't exploit SVE2
- Go compiler may not optimally schedule across 4 engines

**Expected with SVE2 optimization**:
- Better utilization of 4× engines
- Higher IPC (instructions per clock)
- Efficiency closer to 95-98%

## Why Graviton4 is Faster Despite Narrower Vectors

**G3 has 2× wider vectors (256-bit vs 128-bit), so why is G4 faster?**

### Architectural Improvements:
1. **Better IPC**: Neoverse V2 has wider out-of-order window
2. **Branch prediction**: More sophisticated predictor
3. **Memory subsystem**: DDR5 + improved prefetchers
4. **More vector units**: 4× 128-bit > 2× 256-bit for some workloads
5. **SVE2 ISA**: Better instructions (even unused, base ISA improved)

### What This Means:
- V2 core is simply **better** than V1
- Vector width is only one factor
- IPC and memory access matter more for this workload

## Recommendations

### For Maximum Single-Thread Performance: ✅ **Graviton4**
- 18% faster than Graviton3
- Best if: Single-threaded, compute-intensive

### For Maximum Parallel Performance: ✅ **Graviton4**
- 7% faster than Graviton3 at 8 cores
- 50% more cores available (c8g.16xlarge has 64 vCPUs vs G3's 48)

### For Best Efficiency: ✅ **Graviton3**
- 98.6% parallel efficiency
- If scaling to many cores, efficiency matters

### For Best Price/Performance: ✅ **Graviton3** (probably)
- Need to verify c8g pricing
- If G4 premium < 7%, then G4 wins
- If G4 premium > 7%, then G3 wins

## Optimization Opportunities

### Current: Pure Go (No SIMD)
- Compiler-generated code only
- Not exploiting SVE or SVE2
- **Result**: 0.26 GFLOPS single-thread

### With NEON Assembly:
- **Expected**: 0.5-1.0 GFLOPS (2-4× improvement)
- **Effort**: 8h (port from macOS)
- **Benefit**: Works on all Graviton

### With SVE2 Intrinsics:
- **Expected**: 1.0-1.5 GFLOPS (4-6× improvement)
- **Effort**: 16h (new implementation)
- **Benefit**: Graviton4-specific, exploits 4× engines

### With OpenBLAS: ⭐ **RECOMMENDED**
- **Expected**: 2.5-5.0 GFLOPS (10-20× improvement)
- **Effort**: 4h (CGo bindings)
- **Benefit**: Works on all Graviton, battle-tested

## Next Steps

1. **Terminate G4 instance** (save costs)
2. **Implement OpenBLAS support** (highest ROI)
3. **Re-benchmark with OpenBLAS** (expect 10-20× speedup)
4. **Test on larger instances** (c8g.16xlarge with 64 vCPUs)

## Files

- `graviton4_log.txt` - Quick benchmark results
- `graviton4_full_log.txt` - Full benchmark (4 sizes)
- `graviton4.json` - Quick benchmark JSON
- `graviton4_full.json` - Full benchmark JSON
- `GRAVITON4_ANALYSIS.md` - This file

## Conclusion

**Graviton4 is the fastest Graviton processor** across all metrics, despite having narrower vectors than Graviton3. The Neoverse V2 core architecture improvements (IPC, memory, branch prediction) outweigh the vector width disadvantage.

**Parallel efficiency trade-off**: G4 sacrifices some efficiency (89% vs G3's 99%) but still delivers 7% more absolute performance. This improves with larger matrices, suggesting thread coordination overhead rather than memory bandwidth limits.

**Biggest opportunity**: OpenBLAS integration would give 10-20× speedup for ~4 hours effort, far better ROI than hand-coded SIMD.
