#!/bin/bash
# Working Graviton Test Script
# Based on manual testing on Graviton3

set -e

GENERATION=${1:-graviton3}
AWS_PROFILE="${AWS_PROFILE:-aws}"
AWS_REGION="${AWS_REGION:-us-west-2}"

AMI_ID="ami-0c5777a14602ab4b9"
KEY_NAME="aws-benchmark-test"
SECURITY_GROUP="sg-5059b179"
KEY_FILE="$HOME/.ssh/aws-benchmark-test.pem"

case "$GENERATION" in
    graviton2) INSTANCE_TYPE="c6g.xlarge"; NAME="graviton2-final" ;;
    graviton3) INSTANCE_TYPE="c7g.xlarge"; NAME="graviton3-final" ;;
    *) echo "Usage: $0 graviton2|graviton3"; exit 1 ;;
esac

echo "=== Launching $NAME ($INSTANCE_TYPE) ==="

# Launch instance
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

# Install dependencies
echo "=== Installing dependencies ==="
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP <<'REMOTE'
# Install gcc (Go package manager will fail, so we install from tarball)
sudo dnf install -y gcc

# Install Go 1.25.1 from official tarball
cd /tmp
wget -q https://go.dev/dl/go1.25.1.linux-arm64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.25.1.linux-arm64.tar.gz
/usr/local/go/bin/go version
REMOTE

# Create and upload simple CPU test
echo "=== Creating CPU test ==="
cat > /tmp/test_cpu.go <<'EOF'
package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	data, _ := os.ReadFile("/proc/cpuinfo")
	cpuinfo := string(data)

	var hasSVE, hasSVE2, part string
	for _, line := range strings.Split(cpuinfo, "\n") {
		if strings.HasPrefix(line, "Features") {
			if strings.Contains(line, " sve ") || strings.Contains(line, " sve\t") {
				hasSVE = "YES"
			}
			if strings.Contains(line, " sve2 ") {
				hasSVE2 = "YES"
			}
		}
		if strings.HasPrefix(line, "CPU part") {
			part = strings.TrimSpace(strings.Split(line, ":")[1])
		}
	}

	fmt.Println("=== CPU Detection ===")
	fmt.Println("CPU Part:", part)

	var gen string
	switch part {
	case "0xd0c":
		gen = "Graviton2 (Neoverse N1)"
	case "0xd40":
		gen = "Graviton3 (Neoverse V1)"
	case "0xd4f":
		gen = "Graviton4 (Neoverse V2)"
	default:
		gen = "Unknown"
	}

	fmt.Println("Generation:", gen)
	fmt.Println("SVE Support:", hasSVE)
	fmt.Println("SVE2 Support:", hasSVE2)
}
EOF

scp -i "$KEY_FILE" -o StrictHostKeyChecking=no /tmp/test_cpu.go ec2-user@$PUBLIC_IP:/tmp/

# Run test
echo "=== Running test ==="
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP '/usr/local/go/bin/go run /tmp/test_cpu.go'

echo ""
echo "=== Summary ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION"
echo ""
echo "$INSTANCE_ID|$NAME|$PUBLIC_IP" >> graviton_instances.txt
