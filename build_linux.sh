#!/bin/bash
# Build script for Linux ARM64 (AWS Graviton)
# Uses Go build tags to automatically exclude macOS-specific code

set -e

echo "=== Building for Linux ARM64 (Graviton) ==="

# Build directly - Go build tags will exclude macOS files automatically
# Files with "//go:build darwin" or "// +build darwin" are automatically excluded
GOOS=linux GOARCH=arm64 go build -o local-code-model-linux .

echo ""
echo "✅ Build complete: local-code-model-linux"
echo ""
echo "Files automatically excluded by build tags:"
echo "  - metal.go (darwin,cgo)"
echo "  - accelerate.go (darwin,cgo)"
echo "  - ane_mpsgraph.m (darwin,cgo)"
echo "  - metal_test.go (darwin,cgo)"
echo "  - accelerate_test.go (darwin,cgo)"
echo "  - cpu_features_darwin.go (darwin)"
echo ""
echo "Files included for Linux:"
echo "  - metal_linux.go (linux) - stubs returning 'macOS only'"
echo "  - accelerate_linux.go (linux) - stubs returning 'macOS only'"
echo "  - cpu_features_linux.go (linux,arm64) - Graviton CPU detection"
echo "  - matmul_sve_linux.go (linux,arm64) - SVE implementation"
echo ""
echo "To test on Graviton:"
echo "  scp -i ~/.ssh/aws-benchmark-test.pem local-code-model-linux ec2-user@<IP>:/tmp/"
echo "  ssh -i ~/.ssh/aws-benchmark-test.pem ec2-user@<IP> '/tmp/local-code-model-linux'"
