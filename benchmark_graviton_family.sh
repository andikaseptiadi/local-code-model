#!/bin/bash
# Comprehensive Graviton Family Benchmarking Script
# Tests Graviton2, Graviton3, and Graviton4 (if available)

set -e

AWS_PROFILE="${AWS_PROFILE:-aws}"
AWS_REGION="${AWS_REGION:-us-west-2}"
AMI_ID="ami-0c5777a14602ab4b9"  # Amazon Linux 2023 ARM64
KEY_NAME="aws-benchmark-test"
SECURITY_GROUP="sg-5059b179"
KEY_FILE="$HOME/.ssh/aws-benchmark-test.pem"

# Graviton instance types
GRAVITON2_INSTANCE="c6g.2xlarge"    # 8 vCPUs, NEON only
GRAVITON3_INSTANCE="c7g.2xlarge"    # 8 vCPUs, SVE 256-bit
GRAVITON3E_INSTANCE="c7gn.2xlarge"  # 8 vCPUs, SVE 256-bit, 35% higher vector perf
GRAVITON4_INSTANCE="c8g.2xlarge"    # 8 vCPUs, SVE2 128-bit (if available)

RESULTS_DIR="graviton_benchmark_results"
mkdir -p "$RESULTS_DIR"

# Build fresh binary
echo "=== Building Linux ARM64 binary ==="
GOOS=linux GOARCH=arm64 go build -o local-code-model-linux .
ls -lh local-code-model-linux

# Function to launch instance
launch_instance() {
    local instance_type=$1
    local name=$2

    echo ""
    echo "=== Launching $name ($instance_type) ==="

    INSTANCE_ID=$(aws ec2 run-instances \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$instance_type" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    echo "Instance ID: $INSTANCE_ID"

    aws ec2 wait instance-running \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID"

    PUBLIC_IP=$(aws ec2 describe-instances \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo "Public IP: $PUBLIC_IP"
    echo "Waiting for SSH..."
    sleep 30

    # Return instance info
    echo "$INSTANCE_ID|$PUBLIC_IP"
}

# Function to run benchmarks on instance
run_benchmark() {
    local ip=$1
    local name=$2
    local json_file=$3
    local log_file=$4

    echo ""
    echo "=== Running benchmark on $name ==="

    # Upload binary
    scp -i "$KEY_FILE" -o StrictHostKeyChecking=no local-code-model-linux ec2-user@$ip:/tmp/

    # Run benchmark
    ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$ip <<'REMOTE' | tee "$log_file"
#!/bin/bash
set -e

echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "CPUs: $(nproc)"
echo ""

echo "=== CPU Detection ==="
grep -E "CPU part|Features" /proc/cpuinfo | head -3
echo ""

# Detect Graviton generation
CPU_PART=$(grep "CPU part" /proc/cpuinfo | head -1 | awk '{print $4}')
case "$CPU_PART" in
    0xd0c) GENERATION="Graviton2 (Neoverse N1)" ;;
    0xd40) GENERATION="Graviton3 (Neoverse V1)" ;;
    0xd4f) GENERATION="Graviton4 (Neoverse V2)" ;;
    *) GENERATION="Unknown ($CPU_PART)" ;;
esac

echo "Detected: $GENERATION"
echo ""

# Check for vector extensions
HAS_SVE=$(grep -o "sve" /proc/cpuinfo | head -1 || echo "")
HAS_SVE2=$(grep -o "sve2" /proc/cpuinfo | head -1 || echo "")

if [ -n "$HAS_SVE2" ]; then
    echo "Vector Support: SVE2 ✅"
elif [ -n "$HAS_SVE" ]; then
    echo "Vector Support: SVE ✅"
else
    echo "Vector Support: NEON only (no SVE)"
fi
echo ""

# Make binary executable
chmod +x /tmp/local-code-model-linux

# Run hardware detection
echo "=== Hardware Detection ==="
/tmp/local-code-model-linux benchmark -detect
echo ""

# Run quick benchmark
echo "=== Quick Benchmark ==="
/tmp/local-code-model-linux benchmark -quick -json=/tmp/benchmark.json -visualize -format=ascii
echo ""

# Show JSON results
echo "=== JSON Results ==="
cat /tmp/benchmark.json
REMOTE

    # Download results
    scp -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$ip:/tmp/benchmark.json "$json_file" 2>/dev/null || true

    echo ""
    echo "Results saved to:"
    echo "  Log: $log_file"
    echo "  JSON: $json_file"
}

# Track launched instances
INSTANCES=()

# Launch Graviton2
echo "=== GRAVITON2 ==="
G2_INFO=$(launch_instance "$GRAVITON2_INSTANCE" "graviton2-bench")
G2_ID=$(echo "$G2_INFO" | cut -d'|' -f1)
G2_IP=$(echo "$G2_INFO" | cut -d'|' -f2)
INSTANCES+=("$G2_ID")

run_benchmark "$G2_IP" "Graviton2" \
    "$RESULTS_DIR/graviton2_results.json" \
    "$RESULTS_DIR/graviton2_log.txt"

# Launch Graviton3
echo "=== GRAVITON3 ==="
G3_INFO=$(launch_instance "$GRAVITON3_INSTANCE" "graviton3-bench")
G3_ID=$(echo "$G3_INFO" | cut -d'|' -f1)
G3_IP=$(echo "$G3_INFO" | cut -d'|' -f2)
INSTANCES+=("$G3_ID")

run_benchmark "$G3_IP" "Graviton3" \
    "$RESULTS_DIR/graviton3_results.json" \
    "$RESULTS_DIR/graviton3_log.txt"

# Launch Graviton3E
echo "=== GRAVITON3E ==="
G3E_INFO=$(launch_instance "$GRAVITON3E_INSTANCE" "graviton3e-bench")
G3E_ID=$(echo "$G3E_INFO" | cut -d'|' -f1)
G3E_IP=$(echo "$G3E_INFO" | cut -d'|' -f2)
INSTANCES+=("$G3E_ID")

run_benchmark "$G3E_IP" "Graviton3E" \
    "$RESULTS_DIR/graviton3e_results.json" \
    "$RESULTS_DIR/graviton3e_log.txt"

# Try Graviton4 (may not be available in all regions)
echo "=== GRAVITON4 ==="
echo "Attempting to launch Graviton4 instance..."
if G4_INFO=$(launch_instance "$GRAVITON4_INSTANCE" "graviton4-bench" 2>/dev/null); then
    G4_ID=$(echo "$G4_INFO" | cut -d'|' -f1)
    G4_IP=$(echo "$G4_INFO" | cut -d'|' -f2)
    INSTANCES+=("$G4_ID")

    run_benchmark "$G4_IP" "Graviton4" \
        "$RESULTS_DIR/graviton4_results.json" \
        "$RESULTS_DIR/graviton4_log.txt"
else
    echo "⚠️  Graviton4 instances not available in $AWS_REGION"
    echo "    (c8g instances are in limited preview)"
fi

# Generate comparison report
echo ""
echo "=== Generating Comparison Report ==="

cat > "$RESULTS_DIR/COMPARISON.md" <<'EOF'
# Graviton Family Benchmark Comparison

## Test Configuration

- **Date**: $(date -u +"%Y-%m-%d %H:%M UTC")
- **Region**: $AWS_REGION
- **Instance Types**:
  - Graviton2: $GRAVITON2_INSTANCE (8 vCPUs)
  - Graviton3: $GRAVITON3_INSTANCE (8 vCPUs)
  - Graviton3E: $GRAVITON3E_INSTANCE (8 vCPUs)
  - Graviton4: $GRAVITON4_INSTANCE (8 vCPUs)

## Architecture Summary

| Generation | CPU | Vector Extension | Vector Width | Cores/vCPUs |
|------------|-----|------------------|--------------|-------------|
| Graviton2 | Neoverse N1 | NEON only | 128-bit | 64 |
| Graviton3 | Neoverse V1 | SVE | 256-bit | 64 |
| Graviton3E | Neoverse V1 | SVE | 256-bit | 64 |
| Graviton4 | Neoverse V2 | SVE2 | 128-bit | 96 |

## Results

See individual files for detailed results:
- `graviton2_log.txt` - Graviton2 full output
- `graviton2_results.json` - Graviton2 JSON data
- `graviton3_log.txt` - Graviton3 full output
- `graviton3_results.json` - Graviton3 JSON data
- `graviton3e_log.txt` - Graviton3E full output
- `graviton3e_results.json` - Graviton3E JSON data
- `graviton4_log.txt` - Graviton4 full output (if available)
- `graviton4_results.json` - Graviton4 JSON data (if available)

## Key Findings

### Vector Width Impact

Graviton3 has **2× wider vectors** (256-bit) than Graviton2 (128-bit) and Graviton4 (128-bit).

Expected performance for vector operations:
- Graviton2: Baseline (NEON 128-bit)
- Graviton3: ~2× faster (SVE 256-bit)
- Graviton3E: ~2.7× faster (SVE 256-bit + 35% higher vector perf)
- Graviton4: ~1.3-1.5× faster (SVE2 128-bit, better ISA)

### Multi-Threading Impact

Graviton4 has **50% more cores** (96 vs 64) but narrower vectors.

Expected performance for parallel workloads:
- Single-threaded: Graviton3 wins (wider vectors)
- Multi-threaded: Graviton4 wins (more cores)

## Recommendations

Based on workload characteristics:

### Choose Graviton2 If:
- Legacy code compatibility needed
- Budget constrained
- Workload not vector-heavy

### Choose Graviton3/3E If:
- Heavily vectorized single-threaded code
- Vector search, signal processing
- Maximum per-core vector throughput

### Choose Graviton4 If:
- Highly parallel workloads
- More threads > wider vectors
- Need latest SVE2 ISA features
- Memory bandwidth critical (DDR5)

## Cost Efficiency

| Instance | $/hour | Relative Cost | Best For |
|----------|--------|---------------|----------|
| c6g.2xlarge | ~$0.27 | Baseline | General compute |
| c7g.2xlarge | ~$0.29 | +7% | Vector workloads |
| c8g.2xlarge | TBD | TBD | Parallel workloads |

## Next Steps

1. Analyze JSON results for GFLOPS comparison
2. Test with OpenBLAS integration
3. Compare against macOS (M4 Max) results
4. Test on larger instances (16xlarge) for scaling
EOF

echo "Comparison report: $RESULTS_DIR/COMPARISON.md"

# Summary
echo ""
echo "=== BENCHMARK COMPLETE ==="
echo ""
echo "Results directory: $RESULTS_DIR/"
echo ""
echo "Launched instances:"
for id in "${INSTANCES[@]}"; do
    echo "  - $id"
done
echo ""
echo "To terminate all instances:"
echo "  aws ec2 terminate-instances --instance-ids ${INSTANCES[*]} --profile $AWS_PROFILE --region $AWS_REGION"
echo ""
echo "Instances saved to: $RESULTS_DIR/instances.txt"
echo "${INSTANCES[@]}" > "$RESULTS_DIR/instances.txt"
