#!/bin/bash
# ===========================================================================
# Direct Graviton Testing Script
# ===========================================================================
#
# This script creates a tarball of the code, launches an instance,
# uploads code, runs tests, and retrieves results.
#
# Usage: ./test_on_graviton.sh graviton2|graviton3
#
# ===========================================================================

set -e

GENERATION=${1:-graviton2}
AWS_PROFILE="${AWS_PROFILE:-aws}"
AWS_REGION="${AWS_REGION:-us-west-2}"
KEY_FILE="$HOME/.ssh/aws-benchmark-test.pem"

# Configuration
AMI_ID="ami-0c5777a14602ab4b9"  # AL2023 ARM64
KEY_NAME="aws-benchmark-test"
SECURITY_GROUP="sg-5059b179"

# Instance types
case "$GENERATION" in
    graviton2)
        INSTANCE_TYPE="c6g.xlarge"
        NAME="graviton2-benchmark"
        ;;
    graviton3)
        INSTANCE_TYPE="c7g.xlarge"
        NAME="graviton3-benchmark"
        ;;
    *)
        echo "Usage: $0 graviton2|graviton3"
        exit 1
        ;;
esac

echo "=== Testing on $NAME ($INSTANCE_TYPE) ==="

# Create tarball of code (from repo root)
echo "Creating code tarball..."
cd /Users/scttfrdmn/src/local-code-model
tar czf /tmp/local-code-model.tar.gz \
    --exclude='.git' \
    --exclude='*.o' \
    --exclude='*.test' \
    --exclude='local-code-model' \
    .

# Launch instance
echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance running at: $PUBLIC_IP"
echo "Waiting 30s for SSH to be ready..."
sleep 30

# Upload code
echo "Uploading code..."
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    /tmp/local-code-model.tar.gz ec2-user@$PUBLIC_IP:/tmp/

# Start tmux session and run tests
echo "Starting tmux session and running tests..."
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ec2-user@$PUBLIC_IP <<'ENDSSH'
# Start detached tmux session
tmux new-session -d -s graviton-test 'bash -x /tmp/run_tests.sh'
echo "Tests running in tmux session 'graviton-test'"
echo "To attach: tmux attach -t graviton-test"
ENDSSH

# Upload test script
echo "Uploading test script..."
cat > /tmp/run_tests.sh <<'TESTSCRIPT'
#!/bin/bash
set -x
cd /tmp
mkdir -p graviton-test
cd graviton-test
tar xzf /tmp/local-code-model.tar.gz

# Install dependencies
sudo dnf install -y golang gcc openblas-devel 2>&1 | tail -5

# Show CPU info
echo "=== CPU Information ==="
cat /proc/cpuinfo | grep -E "(processor|implementer|architecture|variant|part|revision|Features)" | head -20

# Build
echo "=== Building ==="
go build -tags linux,arm64,cgo .

# CPU Detection
echo "=== CPU Detection ==="
go test -v -run TestCPUFeatureDetection 2>&1
go test -v -run TestGravitonDetection 2>&1

# Correctness
echo "=== Correctness Tests ==="
go test -v -run TestSVEMatMulCorrectness 2>&1 || echo "SVE not available (expected on Graviton2)"

# Benchmarks
echo "=== Benchmarks ==="
go test -bench BenchmarkGravitonComparison -benchtime=3s -run=^$ 2>&1
echo ""
go test -bench BenchmarkGravitonNEON -benchtime=3s -run=^$ 2>&1 || true
echo ""
go test -bench BenchmarkGravitonSVE -benchtime=3s -run=^$ 2>&1 || true

echo "=== Complete ===" | tee /tmp/graviton-results.txt
TESTSCRIPT

scp -i "$KEY_FILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    /tmp/run_tests.sh ec2-user@$PUBLIC_IP:/tmp/
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ec2-user@$PUBLIC_IP 'chmod +x /tmp/run_tests.sh && tmux new-session -d -s graviton-test "bash -x /tmp/run_tests.sh 2>&1 | tee /tmp/graviton-results.txt"'

# Save instance info for later cleanup
echo "$INSTANCE_ID|$NAME|$PUBLIC_IP" >> graviton_instances.txt

echo ""
echo "=== Instance Launched ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Tests are running in tmux. Wait 2-3 minutes, then retrieve results:"
echo "  scp -i $KEY_FILE ec2-user@$PUBLIC_IP:/tmp/graviton-results.txt ./$NAME-results.txt"
echo ""
echo "To SSH and check: ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo "  tmux attach -t graviton-test"
echo ""
echo "To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION"
echo ""
