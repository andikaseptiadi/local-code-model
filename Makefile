# Makefile for local-code-model
# Provides targets for Go Report Card compliance checks

.PHONY: help fmt vet staticcheck lint quality build test clean install-tools

# Default target: show help
help:
	@echo "Local Code Model - Makefile"
	@echo ""
	@echo "Quality Checks (Go Report Card compliance):"
	@echo "  make fmt          - Check code formatting with gofmt"
	@echo "  make vet          - Run go vet static analysis"
	@echo "  make staticcheck  - Run staticcheck linter"
	@echo "  make lint         - Run golangci-lint"
	@echo "  make quality      - Run all quality checks"
	@echo ""
	@echo "Build & Test:"
	@echo "  make build        - Build the project"
	@echo "  make test         - Run all tests"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "Setup:"
	@echo "  make install-tools - Install required linting tools"

# Check code formatting with gofmt
fmt:
	@echo "==> Running gofmt..."
	@gofmt -l . | tee /dev/stderr | (! grep .)
	@echo "✓ gofmt check passed"

# Run go vet
vet:
	@echo "==> Running go vet..."
	@go vet ./...
	@echo "✓ go vet passed"

# Run staticcheck
staticcheck:
	@echo "==> Running staticcheck..."
	@command -v staticcheck >/dev/null 2>&1 || { echo "staticcheck not found. Run 'make install-tools'"; exit 1; }
	@staticcheck ./...
	@echo "✓ staticcheck passed"

# Run golangci-lint with additional Go Report Card checks
# (gofmt, govet, staticcheck are already run separately)
lint:
	@echo "==> Running golangci-lint (ineffassign, misspell, gocyclo)..."
	@command -v golangci-lint >/dev/null 2>&1 || { echo "golangci-lint not found. Run 'make install-tools'"; exit 1; }
	@golangci-lint run --enable=ineffassign,misspell,gocyclo --timeout=5m
	@echo "✓ golangci-lint passed"

# Run all quality checks (Go Report Card compliance)
quality: fmt vet staticcheck lint
	@echo ""
	@echo "===================================="
	@echo "✓ All quality checks passed!"
	@echo "===================================="

# Build the project
build:
	@echo "==> Building local-code-model..."
	@go build -o local-code-model
	@echo "✓ Build successful"

# Run tests
test:
	@echo "==> Running tests..."
	@go test ./...
	@echo "✓ Tests passed"

# Clean build artifacts
clean:
	@echo "==> Cleaning..."
	@go clean
	@rm -f local-code-model
	@rm -f *.bin *.txt
	@echo "✓ Cleaned"

# Install required linting tools
install-tools:
	@echo "==> Installing linting tools..."
	@echo "Installing staticcheck..."
	@go install honnef.co/go/tools/cmd/staticcheck@latest
	@echo "Installing golangci-lint..."
	@curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(shell go env GOPATH)/bin latest
	@echo "✓ Tools installed successfully"
