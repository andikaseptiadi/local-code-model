# Benchmarking Guide

This guide explains how to run comprehensive benchmarks on all supported platforms.

## Quick Start

### macOS (Apple Silicon)
```bash
# Benchmarks completed! See benchmark_macos_results.txt
go test -bench . -benchtime=3s -timeout=30m
```

### Jetson Orin NX
```bash
# Copy files to Jetson
scp -r . orin1.local:~/local-code-model/

# SSH to Jetson
ssh orin1.local
cd ~/local-code-model

# Run benchmarks
./run_benchmarks_orin.sh

# Copy results back
scp orin1.local:~/local-code-model/benchmark_orin_results.txt .
```

### AWS Graviton
```bash
# Launch instance (choose generation)
# Graviton2: c6g.16xlarge (64 cores, $2.72/hr)
# Graviton3: c7g.16xlarge (64 cores, $2.90/hr)
# Graviton4: c8g.16xlarge (64 cores, TBD)

# Copy code
scp -r . ec2-user@<instance-ip>:~/local-code-model/

# SSH and run
ssh ec2-user@<instance-ip>
cd ~/local-code-model
./run_benchmarks_graviton.sh

# Copy results back
scp ec2-user@<instance-ip>:~/local-code-model/benchmark_graviton_*.txt .
```

### AWS g5g (Graviton2 + T4)
```bash
# Launch g5g.16xlarge (64 cores, T4 GPU, ~$3/hr)

# Copy code
scp -r . ec2-user@<instance-ip>:~/local-code-model/

# SSH and run
ssh ec2-user@<instance-ip>
cd ~/local-code-model

# Setup CUDA (if not already installed)
source setup_g5g.sh

# Run benchmarks
./run_benchmarks_g5g.sh

# Copy results back
scp ec2-user@<instance-ip>:~/local-code-model/benchmark_g5g_results.txt .
```

---

## Platform-Specific Details

### macOS M4 Max

**Benchmarks Completed:** ✅ October 1, 2025

**Results Summary:**
- **Accelerate (Apple BLAS):** 702 GFLOPS (FP64), 1114 GFLOPS (FP32)
- **Metal GPU:** 770 GFLOPS @ 1024×1024
- **ANE:** 735 GFLOPS (INT8-optimized)
- **Speedup:** 1368× vs naive implementation

**Transformer Performance:**
- Forward pass: 103ms (256 dim, 128 seq, 4 layers)
- Attention: 18ms
- Feed-forward: 24ms

**File:** `benchmark_macos_results.txt` (87 lines, full results)

---

### Jetson Orin NX

**Status:** ⏳ Pending hardware access

**Expected Results:**
- OpenBLAS: 92 GFLOPS (from earlier testing)
- CUDA: 23 GFLOPS (FP64 limited)
- NEON: 0.93 GFLOPS (slower than naive!)

**What to Test:**
1. Confirm OpenBLAS dominance over GPU for FP64
2. Verify NEON performance regression
3. Test CUDA scaling (64 → 4096)
4. ARM Compute Library (if installed)

**How to Run:**
```bash
# On Jetson
./run_benchmarks_orin.sh

# Examines:
# - CUDA version and GPU specs
# - All matrix multiplication backends
# - Transformer forward pass
# - GFLOPS calculations
```

**Expected Runtime:** ~6-8 minutes

---

### AWS Graviton

**Status:** ⏳ Pending instance launch

#### Graviton2 (c6g instances)

**Hardware:**
- CPU: Neoverse N1 (64 cores @ 2.5 GHz)
- Vector: NEON only (128-bit)

**Expected:**
- OpenBLAS: 30-40 GFLOPS (64 cores)
- NEON: ~15-20 GFLOPS if optimized

**Instance Recommendation:** `c6g.16xlarge` (64 cores, $2.72/hr)

#### Graviton3/3E (c7g instances)

**Hardware:**
- CPU: Neoverse V1 (64 cores @ 2.6 GHz)
- Vector: 2× SVE engines (256-bit each)

**Expected:**
- OpenBLAS: 60-80 GFLOPS (SVE benefit)
- SVE: Should see 2× improvement over NEON

**Instance Recommendation:** `c7g.16xlarge` (64 cores, $2.90/hr)

#### Graviton4 (c8g instances)

**Hardware:**
- CPU: Neoverse V2 (48-96 cores @ 2.8 GHz)
- Vector: 4× SVE2 engines (128-bit each)

**Expected:**
- OpenBLAS: 80-100 GFLOPS (4× SVE2 engines)
- SVE2: Best per-core performance

**Instance Recommendation:** `c8g.16xlarge` (64 cores, pricing TBD)

**Testing Script:**

The `run_benchmarks_graviton.sh` script:
1. Auto-detects Graviton generation (N1/V1/V2)
2. Checks for SVE/SVE2 support
3. Runs full benchmark suite
4. Saves results to `benchmark_graviton_<generation>_results.txt`

**Key Comparisons:**
- OpenBLAS scaling across generations
- SVE vs NEON performance delta
- Multi-engine utilization (G4's 4× units)

---

### AWS g5g (Graviton2 + T4)

**Status:** ⏳ Pending instance launch

**Hardware:**
- CPU: Graviton2 (Neoverse N1, 64 cores)
- GPU: NVIDIA T4 (40 SMs, 2560 CUDA cores, 16GB VRAM)
- Compute Capability: 7.5

**Expected Results (1024×1024 FP64):**
- CPU (OpenBLAS): 30-40 GFLOPS
- GPU (T4): 150-200 GFLOPS ✅ Should win

**Why T4 Should Outperform Orin:**
- 40 SMs vs Orin's 8 (5× more compute)
- Dedicated VRAM (no unified memory overhead)
- Higher clock speeds (1590 MHz boost)

**Critical Tests:**
1. CPU vs GPU comparison (BenchmarkG5GComparison)
2. CUDA scaling (64 → 4096)
3. FP32 vs FP64 performance delta (should be 64×)
4. Data transfer overhead analysis

**Instance Recommendation:** `g5g.16xlarge` (64 cores, T4, ~$3/hr)

**Setup:**
```bash
# Use Amazon Linux AMI with NVIDIA drivers pre-installed
# Or run setup script:
source setup_g5g.sh
```

---

## Benchmark Analysis

### What Each Benchmark Tests

#### Matrix Multiplication Backends
- **BenchmarkAccelerateVsCPU:** Naive vs Accelerate (macOS only)
- **BenchmarkG5GComparison:** Compare all available backends (CPU, GPU, SIMD)
- **BenchmarkCUDAScaling:** GPU performance scaling (64 → 4096)
- **BenchmarkSVEGravitonComparison:** SVE vs NEON (Graviton only)

#### Cache and Memory
- **BenchmarkCacheBlockSizes:** Optimal block size (16, 32, 64, 128, 256)
- **BenchmarkMatMulStrategies:** Naive vs Parallel vs Cached
- **BenchmarkMatMulWorkerCounts:** Scaling with thread count (1, 2, 4, 8, ...)

#### Transformer Components
- **BenchmarkAttention:** Multi-head self-attention
- **BenchmarkTransformerBlock:** Full transformer block
- **BenchmarkGPTForward:** Complete model forward pass

#### Platform-Specific
- **BenchmarkMetalVsCPU:** Metal GPU vs CPU (macOS)
- **BenchmarkANEMPSGraph:** Neural Engine performance (macOS)
- **BenchmarkComprehensive:** All backends side-by-side

### Interpreting Results

#### GFLOPS Calculation
For matrix multiplication (M×K) @ (K×N):
```
FLOPs = 2 * M * K * N
GFLOPS = FLOPs / time_in_seconds / 1e9
```

Example (1024×1024):
```
FLOPs = 2 * 1024 * 1024 * 1024 = 2,147,483,648
Time = 3,061,559 ns = 0.003061559 seconds
GFLOPS = 2,147,483,648 / 0.003061559 / 1e9 = 701.6
```

#### Speedup vs Naive
```
Speedup = time_naive / time_optimized
```

Example:
```
Naive: 4,191,425,375 ns
Accelerate: 3,061,559 ns
Speedup = 4,191,425,375 / 3,061,559 = 1368×
```

#### Efficiency vs Theoretical Peak

For GPUs, compare achieved GFLOPS to theoretical:
```
Efficiency = (measured_gflops / theoretical_peak) * 100%
```

Example (Jetson Orin, 1024 FP64):
```
Measured: 23 GFLOPS
Theoretical: 250 GFLOPS (FP64)
Efficiency: 23 / 250 = 9.2%
```

Low efficiency indicates:
- Problem too small to saturate GPU
- Memory bandwidth bottleneck
- Data transfer overhead
- FP64 limitations (expected for consumer GPUs)

---

## Expected Total Results

Once all benchmarks are run, we'll have:

### macOS M4 Max
✅ **Complete:** benchmark_macos_results.txt (87 lines)
- 85 benchmarks covering all Apple backends

### Jetson Orin NX
⏳ **Pending:** benchmark_orin_results.txt
- Expected: 60+ benchmarks (CUDA, OpenBLAS, NEON)
- Runtime: ~6-8 minutes
- Key metric: OpenBLAS vs CUDA comparison

### Graviton2
⏳ **Pending:** benchmark_graviton_Graviton2_results.txt
- Expected: 50+ benchmarks (OpenBLAS, NEON)
- Runtime: ~5-7 minutes
- Key metric: Baseline NEON performance

### Graviton3/3E
⏳ **Pending:** benchmark_graviton_Graviton3_results.txt
- Expected: 55+ benchmarks (OpenBLAS, SVE, NEON)
- Runtime: ~5-7 minutes
- Key metric: SVE improvement over G2

### Graviton4
⏳ **Pending:** benchmark_graviton_Graviton4_results.txt
- Expected: 55+ benchmarks (OpenBLAS, SVE2, NEON)
- Runtime: ~5-7 minutes
- Key metric: 4× SVE2 engine utilization

### g5g (T4)
⏳ **Pending:** benchmark_g5g_results.txt
- Expected: 70+ benchmarks (CUDA, OpenBLAS, NEON)
- Runtime: ~8-10 minutes
- Key metric: T4 vs Graviton2 CPU comparison

---

## Troubleshooting

### Build Failures

#### "CUDA not found"
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### "OpenBLAS not found"
```bash
# Debian/Ubuntu
sudo apt-get install libopenblas-dev

# Amazon Linux
sudo yum install openblas-devel
```

#### "ARM Compute Library not found"
```bash
sudo apt-get install libarmcl-dev
```

### Runtime Errors

#### GPU tests fail
- Check `nvidia-smi` for GPU availability
- Verify CUDA compatibility with GPU
- Check unified memory support (Jetson) or VRAM (T4)

#### Benchmarks timeout
- Increase timeout: `-timeout=60m`
- Reduce benchtime: `-benchtime=1s`
- Run specific benchmarks: `-bench BenchmarkG5GComparison`

#### Out of memory
- Reduce matrix sizes in tests
- Skip large matrix tests (2048×2048+)
- Check available memory with `free -h`

---

## Next Steps After Benchmark Completion

1. **Update BENCHMARK_RESULTS.md** with actual results
2. **Compare across platforms:**
   - CPU scaling (M4 vs Orin vs Graviton)
   - GPU performance (Metal vs CUDA on Orin vs T4)
   - Vector extensions (NEON vs SVE vs SVE2)
3. **Identify optimization opportunities:**
   - Where does naive beat optimized? (unexpected)
   - What's the memory bottleneck?
   - How well do backends scale?
4. **Document lessons learned:**
   - Update PERFORMANCE_LESSONS.md
   - Add platform-specific insights
   - Recommend best practices

---

## Cost Estimates

### AWS Instances (per hour)

| Instance | Type | vCPUs | Memory | Cost/hr | Recommended Runtime |
|----------|------|-------|--------|---------|---------------------|
| c6g.16xlarge | Graviton2 | 64 | 128 GB | $2.72 | 0.2 hr ($0.54) |
| c7g.16xlarge | Graviton3 | 64 | 128 GB | $2.90 | 0.2 hr ($0.58) |
| c8g.16xlarge | Graviton4 | 64 | 128 GB | TBD | 0.2 hr (TBD) |
| g5g.16xlarge | G2 + T4 | 64 | 128 GB | ~$3.00 | 0.25 hr ($0.75) |

**Total estimated cost:** ~$2-3 for all AWS benchmarks

### Jetson Orin NX
- Already owned/available (orin1.local)
- No additional cost

---

**Last Updated:** October 1, 2025
