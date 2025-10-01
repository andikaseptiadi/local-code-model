#!/bin/bash

# ===========================================================================
# AWS Multi-Architecture Benchmark Runner
# ===========================================================================
#
# This script automates running benchmarks across different AWS Graviton
# instance types to compare ARM architecture performance.
#
# USAGE:
#   ./scripts/run_aws_benchmarks.sh [options]
#
# REQUIREMENTS:
#   - AWS CLI configured with appropriate credentials
#   - SSH key pair for EC2 instances
#   - VPC and security group configured
#
# INSTANCE TYPES TESTED:
#   - c6g.xlarge:  Graviton2 (Neoverse N1, 4 vCPU, DDR4)
#   - c7g.xlarge:  Graviton3 (Neoverse V1, 4 vCPU, DDR5, SVE)
#   - c7gn.xlarge: Graviton3E (Neoverse V1, enhanced networking)
#   - c8g.xlarge:  Graviton4 (Neoverse V2, 4 vCPU, DDR5, SVE2) [when available]
#   - g5g.xlarge:  Graviton2 + NVIDIA T4G GPU
#
# ===========================================================================

set -euo pipefail

# Configuration
AWS_REGION=${AWS_REGION:-us-west-2}
AWS_PROFILE=${AWS_PROFILE:-default}
KEY_NAME=${KEY_NAME:-""}
SECURITY_GROUP=${SECURITY_GROUP:-""}
SUBNET=${SUBNET:-""}
AMI_ID=${AMI_ID:-""}  # Amazon Linux 2023 ARM64

# Output directory
OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

# Check requirements
check_requirements() {
    log "Checking requirements..."

    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install: https://aws.amazon.com/cli/"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        warn "jq not found. Install for better output formatting."
    fi

    # Test AWS credentials
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        error "AWS credentials not configured for profile: $AWS_PROFILE"
        exit 1
    fi

    log "✓ Requirements met"
}

# Detect latest ARM AMI
detect_ami() {
    if [ -z "$AMI_ID" ]; then
        log "Detecting latest Amazon Linux 2023 ARM64 AMI..."
        AMI_ID=$(aws ec2 describe-images \
            --region "$AWS_REGION" \
            --profile "$AWS_PROFILE" \
            --owners amazon \
            --filters \
                "Name=name,Values=al2023-ami-2023*-arm64" \
                "Name=state,Values=available" \
            --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
            --output text)

        log "Using AMI: $AMI_ID"
    fi
}

# Launch an instance
launch_instance() {
    local instance_type=$1
    local name=$2

    log "Launching $instance_type instance..."

    local instance_id
    instance_id=$(aws ec2 run-instances \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" \
        --image-id "$AMI_ID" \
        --instance-type "$instance_type" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --subnet-id "$SUBNET" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    log "Instance launched: $instance_id"
    echo "$instance_id"
}

# Wait for instance to be running
wait_for_instance() {
    local instance_id=$1

    log "Waiting for instance $instance_id to be running..."

    aws ec2 wait instance-running \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" \
        --instance-ids "$instance_id"

    # Wait a bit more for SSH to be ready
    sleep 30

    log "✓ Instance running"
}

# Get instance IP
get_instance_ip() {
    local instance_id=$1

    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text
}

# Run benchmark on instance
run_benchmark() {
    local ip=$1
    local name=$2
    local has_gpu=${3:-false}

    log "Running benchmark on $name ($ip)..."

    # Copy code to instance
    log "Copying code..."
    scp -o StrictHostKeyChecking=no -r \
        ./*.go \
        "ec2-user@$ip:~/"

    # Install Go if needed and run benchmark
    log "Installing Go and running benchmark..."

    local gpu_flag=""
    if [ "$has_gpu" = "true" ]; then
        gpu_flag="-gpu"
    fi

    ssh -o StrictHostKeyChecking=no "ec2-user@$ip" << EOF
        # Install Go
        if ! command -v go &> /dev/null; then
            echo "Installing Go..."
            wget -q https://go.dev/dl/go1.21.5.linux-arm64.tar.gz
            sudo tar -C /usr/local -xzf go1.21.5.linux-arm64.tar.gz
            export PATH=\$PATH:/usr/local/go/bin
        fi

        # Run benchmark
        echo "Running benchmark..."
        /usr/local/go/bin/go run . benchmark -json=results.json $gpu_flag -visualize -format=ascii

        # Show results
        cat results.json
EOF

    # Download results
    mkdir -p "$OUTPUT_DIR"
    scp "ec2-user@$ip:~/results.json" "$OUTPUT_DIR/${name}_${TIMESTAMP}.json"

    log "✓ Benchmark complete for $name"
}

# Terminate instance
terminate_instance() {
    local instance_id=$1

    log "Terminating instance $instance_id..."

    aws ec2 terminate-instances \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" \
        --instance-ids "$instance_id" \
        > /dev/null

    log "✓ Instance terminated"
}

# Benchmark a single architecture
benchmark_architecture() {
    local instance_type=$1
    local name=$2
    local has_gpu=${3:-false}

    log ""
    log "=========================================="
    log "Benchmarking: $name ($instance_type)"
    log "=========================================="

    local instance_id
    instance_id=$(launch_instance "$instance_type" "benchmark-$name")

    wait_for_instance "$instance_id"

    local ip
    ip=$(get_instance_ip "$instance_id")

    run_benchmark "$ip" "$name" "$has_gpu"

    terminate_instance "$instance_id"

    log "✓ $name complete"
}

# Main execution
main() {
    log "AWS Multi-Architecture Benchmark Runner"
    log "========================================"
    log ""
    log "AWS Region: $AWS_REGION"
    log "AWS Profile: $AWS_PROFILE"
    log "Output Dir: $OUTPUT_DIR"
    log ""

    check_requirements
    detect_ami

    # Check if we have required parameters
    if [ -z "$KEY_NAME" ] || [ -z "$SECURITY_GROUP" ] || [ -z "$SUBNET" ]; then
        error "Missing required parameters:"
        echo "  KEY_NAME: SSH key pair name"
        echo "  SECURITY_GROUP: Security group ID (must allow SSH)"
        echo "  SUBNET: Subnet ID"
        echo ""
        echo "Example:"
        echo "  KEY_NAME=my-key SECURITY_GROUP=sg-xxx SUBNET=subnet-xxx ./scripts/run_aws_benchmarks.sh"
        exit 1
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Benchmark each architecture
    log "Starting benchmark suite..."

    # Graviton2
    benchmark_architecture "c6g.xlarge" "graviton2" false

    # Graviton3
    benchmark_architecture "c7g.xlarge" "graviton3" false

    # Graviton3E (if available in region)
    if aws ec2 describe-instance-types \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" \
        --instance-types c7gn.xlarge &> /dev/null; then
        benchmark_architecture "c7gn.xlarge" "graviton3e" false
    else
        warn "c7gn (Graviton3E) not available in $AWS_REGION"
    fi

    # Graviton4 (when available)
    if aws ec2 describe-instance-types \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" \
        --instance-types c8g.xlarge &> /dev/null; then
        benchmark_architecture "c8g.xlarge" "graviton4" false
    else
        warn "c8g (Graviton4) not available in $AWS_REGION yet"
    fi

    # g5g (Graviton2 + GPU)
    benchmark_architecture "g5g.xlarge" "g5g" true

    log ""
    log "=========================================="
    log "All benchmarks complete!"
    log "=========================================="
    log ""
    log "Results saved to: $OUTPUT_DIR/"
    log ""
    log "To compare results:"
    log "  go run . compare -files=\"$OUTPUT_DIR/*.json\""
    log ""
    log "To visualize:"
    log "  go run . visualize -files=\"$OUTPUT_DIR/*.json\" -format=gnuplot"
}

# Run main
main "$@"
