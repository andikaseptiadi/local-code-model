# Performance Atlas: Cross-Architecture Benchmarking

This document describes how to use the benchmarking framework to compare matrix multiplication performance across different ARM architectures.

## Overview

The benchmarking system measures performance across optimization levels:

- **Level 0: Naive** - Single-threaded triple loop
- **Level 1: Parallel** - Multi-core goroutines
- **Level 2: Cache-Blocked** - Tiled algorithm for cache locality
- **Level 3: Cache-Blocked + Parallel** - Combined approach
- **Level 4: SIMD** - Vectorized (future)
- **Level 5: GPU** - Metal/CUDA (future)
- **Level 6: ANE** - Neural Engine (future)

## Quick Start

### Run Benchmarks Locally

```bash
# Hardware detection only
./local-code-model benchmark -detect

# Quick benchmark (2 sizes, 3 iterations)
./local-code-model benchmark -quick -visualize -format=ascii

# Full benchmark with JSON output
./local-code-model benchmark -json=results.json

# Custom sizes and iterations
./local-code-model benchmark -sizes=128,256,512,1024 -iterations=10 -json=results.json
```

### Visualize Results

```bash
# ASCII chart (for terminal viewing)
./local-code-model benchmark -quick -visualize -format=ascii

# CSV export (for spreadsheets)
./local-code-model benchmark -csv=results.csv

# Gnuplot scripts (requires gnuplot)
./local-code-model benchmark -visualize -format=gnuplot
```

## AWS Multi-Architecture Testing

### Supported Architectures

| Instance Type | Architecture | Features | Best For |
|--------------|--------------|----------|----------|
| c6g.xlarge | Graviton2 (Neoverse N1) | ARM v8.2, NEON, DDR4 | Cost-effective baseline |
| c7g.xlarge | Graviton3 (Neoverse V1) | ARM v8.4, SVE, DDR5 | Compute-intensive |
| c7gn.xlarge | Graviton3E (Neoverse V1) | + bfloat16, enhanced networking | ML inference |
| c8g.xlarge | Graviton4 (Neoverse V2) | ARM v9, SVE2, DDR5 | Highest performance |
| g5g.xlarge | Graviton2 + NVIDIA T4G | + GPU acceleration | GPU workloads |

### Automated AWS Benchmarking

```bash
# Set AWS credentials
export AWS_PROFILE=your-profile
export AWS_REGION=us-west-2

# Set infrastructure parameters
export KEY_NAME=your-ssh-key
export SECURITY_GROUP=sg-xxxxxxxxx  # Must allow SSH
export SUBNET=subnet-xxxxxxxxx

# Run benchmarks across all Graviton variants
./scripts/run_aws_benchmarks.sh
```

This will:
1. Launch instances for each architecture
2. Install Go and copy code
3. Run benchmarks
4. Download JSON results
5. Terminate instances

Results saved to `benchmark_results/`

### Manual AWS Testing

```bash
# SSH to instance
ssh -i your-key.pem ec2-user@instance-ip

# Install Go
wget https://go.dev/dl/go1.21.5.linux-arm64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-arm64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Copy code (from local machine)
scp -i your-key.pem *.go ec2-user@instance-ip:~/

# Run benchmark
go run . benchmark -json=graviton3.json -visualize -format=ascii

# Download results
scp -i your-key.pem ec2-user@instance-ip:~/graviton3.json .
```

## Expected Performance Patterns

### Graviton2 (Neoverse N1)
- **Caches**: 64 KB L1, 1 MB L2
- **Memory**: DDR4 (~150 GB/s)
- **SIMD**: NEON 128-bit
- **Expected**: Baseline performance, limited by memory bandwidth

### Graviton3 (Neoverse V1)
- **Caches**: 64 KB L1, 1 MB L2, 32-64 MB L3
- **Memory**: DDR5 (~307 GB/s)
- **SIMD**: SVE 256-bit
- **Expected**: 2x memory bandwidth → better parallel scaling

### Graviton3E
- Same as Graviton3
- **Plus**: bfloat16 support for ML workloads
- **Expected**: Similar to Graviton3 for fp64

### Graviton4 (Neoverse V2)
- **Caches**: 64 KB L1, 2 MB L2, 64 MB L3 (larger!)
- **Memory**: DDR5 (~500 GB/s)
- **SIMD**: SVE2 256-bit
- **Expected**: 60% faster than Graviton3, best cache-blocking gains

### M4 Max (Apple Silicon)
- **Caches**: 192 KB L1, 16 MB L2 (largest!)
- **Memory**: Unified (~400 GB/s)
- **Plus**: Metal GPU, ANE
- **Expected**: Best cache-blocking, GPU acceleration available

## Performance Analysis

### Key Metrics

Each benchmark reports:
- **GFLOPS**: Billions of floating-point operations per second
- **Speedup**: Relative to naive single-threaded baseline
- **Parallel Efficiency**: Actual speedup / number of cores

### What To Look For

1. **Naive Performance (Level 0)**
   - Should be similar across architectures (1-3 GFLOPS)
   - Single-core performance

2. **Parallel Speedup (Level 1)**
   - Limited by memory bandwidth
   - Graviton4 > Graviton3 > Graviton2 (bandwidth differences)
   - Efficiency typically 20-50%

3. **Cache-Blocking Gains (Level 2)**
   - Larger caches → bigger gains
   - M4 Max > Graviton4 > Graviton3 > Graviton2
   - Shows cache hierarchy importance

4. **Combined (Level 3)**
   - Best CPU-only performance
   - Total speedup: 10-50x depending on architecture

### Cross-Architecture Comparison

```bash
# After collecting results from multiple architectures
ls benchmark_results/
# graviton2_20250930.json
# graviton3_20250930.json
# graviton4_20250930.json
# m4_max_20250930.json

# Compare (would need comparison tool - future work)
# go run . compare -files="benchmark_results/*.json"
```

## Understanding the Results

### Performance Cliffs

Each optimization level reveals bottlenecks:

**Naive → Parallel**
- Small gain (1.5-3x) = memory bandwidth limited
- Not CPU compute limited

**Parallel → Cache-Blocked**
- Large gain (2-5x) = cache hierarchy matters
- Algorithmic improvement beats hardware

**CPU → GPU** (when implemented)
- Massive gain (50-500x) = specialized hardware wins
- But has overhead (only worth it for large problems)

### Architecture-Specific Insights

**Why Graviton3 beats Graviton2:**
- 2x memory bandwidth (DDR5 vs DDR4)
- Larger L3 cache (32-64 MB)
- SVE vs NEON (256-bit vs 128-bit)

**Why Graviton4 beats Graviton3:**
- ~60% more memory bandwidth (500 vs 307 GB/s)
- 2x larger L2 cache (2 MB vs 1 MB)
- SVE2 improvements

**Why M4 Max excels at cache-blocking:**
- 3x larger L1 cache (192 KB vs 64 KB)
- 16x larger L2 cache (16 MB vs 1 MB)
- Unified memory architecture

## Troubleshooting

### Build Issues

```bash
# Ensure Go is installed
go version

# Should be 1.21 or later
# Clean build
go clean
go build
```

### AWS Issues

```bash
# Check credentials
aws sts get-caller-identity --profile $AWS_PROFILE

# Check instance availability
aws ec2 describe-instance-types \
  --region $AWS_REGION \
  --instance-types c7g.xlarge

# Check security group allows SSH
aws ec2 describe-security-groups \
  --group-ids $SECURITY_GROUP
```

### Performance Issues

If benchmarks are slower than expected:

1. **Check for background processes**: `top` or `htop`
2. **Check thermal throttling**: Sustained load may throttle
3. **Check memory**: Ensure enough RAM for matrix sizes
4. **Check CPU governor**: Should be "performance" not "powersave"

```bash
# On Linux, check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance (requires root)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Output Formats

### JSON Format

```json
{
  "timestamp": "2025-09-30T17:34:06Z",
  "hardware": {
    "os": "linux",
    "arch": "arm64",
    "cpu_model": "AWS Graviton3 (Neoverse V1)",
    "num_cpu": 4,
    "has_sve": true
  },
  "results": [
    {
      "strategy": "Naive",
      "level": 0,
      "size": 512,
      "gflops": 1.23,
      "speedup_vs_naive": 1.0
    }
  ]
}
```

### CSV Format

```csv
Architecture,OS,Arch,Cores,Strategy,Level,Size,AvgTime_ns,GFLOPS,Speedup
Graviton3,linux,arm64,4,Naive,0,512,220000000,1.23,1.00
Graviton3,linux,arm64,4,Parallel,1,512,65000000,4.15,3.37
```

## Future Enhancements

- [ ] SIMD vectorization (ARM NEON/SVE)
- [ ] GPU acceleration (Metal, CUDA)
- [ ] ANE integration (Core ML)
- [ ] Automated comparison tool
- [ ] Web-based visualization dashboard
- [ ] Performance regression testing
- [ ] Power consumption measurements

## Contributing

To add new architectures or optimizations:

1. Add detection in `benchmark.go:DetectHardware()`
2. Add platform characteristics in `cmd_benchmark.go:detectPlatform()`
3. Run benchmarks and document results
4. Submit comparison data

## References

- [AWS Graviton Technical Guide](https://github.com/aws/aws-graviton-getting-started)
- [ARM Neoverse Technical Reference](https://developer.arm.com/Processors/Neoverse)
- [Apple Silicon Performance](https://developer.apple.com/metal/)
- [Matrix Multiplication Optimization](https://arxiv.org/abs/1609.00076)
