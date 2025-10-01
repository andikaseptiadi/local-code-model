#!/bin/bash
# ===========================================================================
# AWS Graviton Testing Script
# ===========================================================================
#
# This script launches AWS Graviton instances and runs benchmarks.
# Usage: ./test_graviton.sh [graviton2|graviton3|graviton4|all]
#
# ===========================================================================

set -e

# Configuration
AWS_PROFILE="${AWS_PROFILE:-aws}"
AWS_REGION="${AWS_REGION:-us-west-2}"
KEY_NAME="${KEY_NAME:-graviton-test-key}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-default}"

# Instance types
GRAVITON2_INSTANCE="c6g.xlarge"    # $0.068/hr - Neoverse N1
GRAVITON3_INSTANCE="c7g.xlarge"    # $0.0725/hr - Neoverse V1
GRAVITON4_INSTANCE="r8g.xlarge"    # Preview - Neoverse V2

# Amazon Linux 2023 ARM64 AMI (update this for your region)
AMI_ID="ami-0e8bb10cbf8f2a2c6"  # AL2023 ARM64 in us-west-2

# User data script to setup and run tests
read -r -d '' USER_DATA <<'EOF' || true
#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=== Graviton Test Setup ==="
date

# Install dependencies
dnf update -y
dnf install -y golang git gcc

# Install OpenBLAS
dnf install -y openblas-devel

# Clone repo (replace with your repo)
cd /home/ec2-user
git clone https://github.com/yourusername/local-code-model.git || true
cd local-code-model

# Build
go build -tags linux,arm64 .

# Run CPU detection tests
echo "=== CPU Detection ==="
go test -v -run TestCPUFeatureDetection
go test -v -run TestGravitonDetection

# Run correctness tests
echo "=== Correctness Tests ==="
go test -v -run TestSVEMatMulCorrectness

# Run benchmarks
echo "=== Benchmarks ==="
go test -bench BenchmarkGravitonComparison -benchtime=5s -run=^$
go test -bench BenchmarkGravitonNEON -benchtime=5s -run=^$
go test -bench BenchmarkGravitonSVE -benchtime=5s -run=^$ || true

# Save results
cp /var/log/user-data.log /home/ec2-user/graviton-results.txt
chown ec2-user:ec2-user /home/ec2-user/graviton-results.txt

echo "=== Complete ==="
date
EOF

# Function to launch instance
launch_instance() {
    local instance_type=$1
    local name=$2

    echo "Launching $name ($instance_type) in $AWS_REGION..."

    instance_id=$(aws ec2 run-instances \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$instance_type" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --user-data "$USER_DATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    echo "Launched instance: $instance_id"
    echo "Waiting for instance to be running..."

    aws ec2 wait instance-running \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id"

    # Get public IP
    public_ip=$(aws ec2 describe-instances \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo "Instance running at: $public_ip"
    echo "Instance ID: $instance_id"
    echo ""
    echo "To SSH: ssh -i ~/.ssh/$KEY_NAME.pem ec2-user@$public_ip"
    echo "To get results: scp -i ~/.ssh/$KEY_NAME.pem ec2-user@$public_ip:~/graviton-results.txt ."
    echo "To terminate: aws ec2 terminate-instances --instance-ids $instance_id --profile $AWS_PROFILE --region $AWS_REGION"
    echo ""

    # Save instance info
    echo "$instance_id|$name|$public_ip" >> graviton_instances.txt
}

# Main
case "${1:-all}" in
    graviton2)
        launch_instance "$GRAVITON2_INSTANCE" "graviton2-test"
        ;;
    graviton3)
        launch_instance "$GRAVITON3_INSTANCE" "graviton3-test"
        ;;
    graviton4)
        launch_instance "$GRAVITON4_INSTANCE" "graviton4-test"
        ;;
    all)
        launch_instance "$GRAVITON2_INSTANCE" "graviton2-test"
        sleep 2
        launch_instance "$GRAVITON3_INSTANCE" "graviton3-test"
        # Graviton4 is preview, may not be available
        # launch_instance "$GRAVITON4_INSTANCE" "graviton4-test"
        ;;
    *)
        echo "Usage: $0 [graviton2|graviton3|graviton4|all]"
        exit 1
        ;;
esac

echo "Instances launched! Check graviton_instances.txt for details."
echo "Wait 5-10 minutes for tests to complete, then fetch results."
