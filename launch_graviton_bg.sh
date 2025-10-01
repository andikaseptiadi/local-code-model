#!/bin/bash
# Background launcher with logging
set -e

GENERATION=${1:-graviton2}
LOG_FILE="graviton-${GENERATION}-$(date +%s).log"

echo "Launching $GENERATION test in background..."
echo "Log file: $LOG_FILE"

nohup bash -c "
set -ex
exec > '$LOG_FILE' 2>&1

AWS_PROFILE=aws
AWS_REGION=us-west-2
AMI_ID='ami-0c5777a14602ab4b9'
KEY_NAME='aws-benchmark-test'
SECURITY_GROUP='sg-5059b179'
KEY_FILE=\$HOME/.ssh/aws-benchmark-test.pem

case '$GENERATION' in
    graviton2) INSTANCE_TYPE='c6g.xlarge'; NAME='graviton2-test' ;;
    graviton3) INSTANCE_TYPE='c7g.xlarge'; NAME='graviton3-test' ;;
esac

echo '=== Creating tarball ==='
cd /Users/scttfrdmn/src/local-code-model
tar czf /tmp/code-\$\$.tar.gz --exclude='.git' --exclude='*.o' .

echo '=== Launching instance ==='
INSTANCE_ID=\$(aws ec2 run-instances \\
    --profile \$AWS_PROFILE \\
    --region \$AWS_REGION \\
    --image-id \$AMI_ID \\
    --instance-type \$INSTANCE_TYPE \\
    --key-name \$KEY_NAME \\
    --security-group-ids \$SECURITY_GROUP \\
    --tag-specifications \"ResourceType=instance,Tags=[{Key=Name,Value=\$NAME}]\" \\
    --query 'Instances[0].InstanceId' \\
    --output text)

echo \"Instance ID: \$INSTANCE_ID\"

aws ec2 wait instance-running \\
    --profile \$AWS_PROFILE \\
    --region \$AWS_REGION \\
    --instance-ids \$INSTANCE_ID

PUBLIC_IP=\$(aws ec2 describe-instances \\
    --profile \$AWS_PROFILE \\
    --region \$AWS_REGION \\
    --instance-ids \$INSTANCE_ID \\
    --query 'Reservations[0].Instances[0].PublicIpAddress' \\
    --output text)

echo \"Public IP: \$PUBLIC_IP\"
echo \"\$INSTANCE_ID|\$NAME|\$PUBLIC_IP\" >> graviton_instances.txt

echo '=== Waiting for SSH ==='
sleep 30

echo '=== Uploading code ==='
scp -i \$KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\
    /tmp/code-\$\$.tar.gz ec2-user@\$PUBLIC_IP:/tmp/code.tar.gz

echo '=== Running tests ==='
ssh -i \$KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\
    ec2-user@\$PUBLIC_IP 'bash -s' <<'REMOTE'
set -x
cd /tmp
mkdir -p test
cd test
tar xzf /tmp/code.tar.gz

echo '=== Installing deps ==='
sudo dnf install -y golang gcc openblas-devel 2>&1 | tail -10

echo '=== CPU Info ==='
cat /proc/cpuinfo | grep -E \"(processor|implementer|part|Features)\" | head -20

echo '=== Building ==='
go build -tags linux,arm64,cgo .

echo '=== CPU Detection ==='
go test -v -run TestCPUFeatureDetection 2>&1
go test -v -run TestGravitonDetection 2>&1

echo '=== Benchmarks ==='
go test -bench BenchmarkGravitonComparison -benchtime=3s -run=^\$ 2>&1

echo '=== COMPLETE ==='
REMOTE

echo \"=== Results saved to $LOG_FILE ===\"
echo \"To terminate: aws ec2 terminate-instances --instance-ids \$INSTANCE_ID --profile \$AWS_PROFILE --region \$AWS_REGION\"
" &

echo "Background process started. Monitor with:"
echo "  tail -f $LOG_FILE"
