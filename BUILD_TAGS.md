# Build Tags and Platform Strategy

## Overview

This project uses Go build tags to ensure **all builds succeed** on all platforms while enabling platform-specific optimizations. This is critical for a teaching resource - students should be able to build and learn on any platform.

## Core Principle

**"Build everywhere, optimize where possible"**

- ✅ macOS builds should succeed
- ✅ Linux builds should succeed
- ✅ Tests should pass on both platforms
- ✅ Platform-specific features gracefully degrade or skip

## Build Tag Patterns

### Pattern 1: Platform-Specific Implementation

**Use case**: Feature only available on one platform (e.g., Metal on macOS)

```go
// metal.go
// +build darwin,cgo

// metal_linux.go
//go:build linux
```

**Result**:
- macOS: Uses real Metal implementation
- Linux: Uses stub that returns helpful error

### Pattern 2: Platform-Specific Tests

**Use case**: Tests for Linux-only features (e.g., Graviton CPU detection)

```go
// graviton_test.go
//go:build linux && arm64
```

**Result**:
- macOS: Test file not compiled, doesn't cause build errors
- Linux ARM64: Test runs and verifies Graviton features
- Linux x86: Test file not compiled

### Pattern 3: Runtime Feature Detection

**Use case**: Optional feature that may or may not exist (e.g., SVE, ANE)

```go
// No build tag needed - always compiled
func NewSVEBackend() (*SVEBackend, error) {
    features := DetectCPUFeatures()
    if !features.HasSVE {
        return &SVEBackend{available: false},
               fmt.Errorf("SVE not available")
    }
    // ... actual implementation
}

// Test with skip
func TestSVE(t *testing.T) {
    backend, err := NewSVEBackend()
    if err != nil {
        t.Skipf("SVE not available: %v", err)
    }
    // ... test SVE features
}
```

**Result**:
- Graviton2: Backend exists but reports unavailable, test skips
- Graviton3/4: Backend works, test runs
- macOS: Backend exists but reports unavailable, test skips

## File Organization

### macOS-Only Files

| File | Build Tag | Purpose |
|------|-----------|---------|
| `metal.go` | `darwin,cgo` | Metal GPU implementation |
| `metal_test.go` | `darwin,cgo` | Metal tests |
| `accelerate.go` | `darwin,cgo` | Accelerate BLAS |
| `accelerate_test.go` | `darwin,cgo` | Accelerate tests |
| `ane_mpsgraph.m` | `darwin,cgo` | ANE Objective-C |
| `ane_mpsgraph_test.go` | `darwin,cgo` | ANE MPSGraph tests |
| `cpu_features_darwin.go` | `darwin` | macOS CPU detection |
| `matmul_simd.go` | `darwin,arm64,!cgo` | NEON assembly (macOS) |
| `matmul_neon_arm64.s` | `darwin,arm64,!cgo` | NEON assembly code |

### Linux-Only Files

| File | Build Tag | Purpose |
|------|-----------|---------|
| `metal_linux.go` | `linux` | Metal/ANE stubs |
| `accelerate_linux.go` | `linux` | Accelerate stub |
| `cpu_features_linux.go` | `linux && arm64` | Graviton detection |
| `cpu_features_test.go` | `linux && arm64` | CPU detection tests |
| `matmul_sve_linux.go` | `linux && arm64` | SVE implementation |
| `matmul_sve.c` | (none) | SVE C intrinsics |
| `matmul_neon_linux.go` | `linux && arm64 && !cgo` | NEON stub |
| `graviton_test.go` | `linux && arm64` | Graviton-specific tests |

### Cross-Platform Files

| File | Build Tag | Purpose |
|------|-----------|---------|
| `tensor.go` | (none) | Core tensor ops |
| `compute.go` | (none) | Backend abstraction |
| `backend.go` | (none) | Backend interface |
| `matmul_optimized.go` | (none) | Portable optimizations |
| `transformer.go` | (none) | Transformer model |
| All other `*.go` | (none) | General code |

## Testing Strategy

### Test Types

1. **Universal Tests** - No build tags, run everywhere
   - `TestTensorBasics`
   - `TestTransformerForward`
   - `TestMatMulCorrectness`

2. **Platform-Specific Tests** - Build tags match implementation
   - `TestMetalMatMul` - Only on `darwin,cgo`
   - `TestGravitonDetection` - Only on `linux && arm64`

3. **Feature-Gated Tests** - Use `t.Skipf()` for unavailable features
   - `TestSVEMatMul` - Skips if SVE not available
   - `TestANE` - Skips if wrong matrix size

### Running Tests

```bash
# macOS - runs macOS tests + universal tests
go test .

# Linux ARM64 - runs Linux tests + universal tests
GOOS=linux GOARCH=arm64 go test .

# Specific platform test
go test . -run TestGraviton  # Only runs on Linux ARM64
go test . -run TestMetal     # Only runs on macOS

# All tests with verbose output
go test . -v
```

## Build Verification

### Local Verification

```bash
# Verify macOS build
go build .
go test .

# Verify Linux ARM64 cross-compile
GOOS=linux GOARCH=arm64 go build .

# Verify Linux x86 cross-compile
GOOS=linux GOARCH=amd64 go build .
```

### CI/CD Strategy

Recommended GitHub Actions:

```yaml
strategy:
  matrix:
    include:
      - os: macos-latest
        goarch: arm64
      - os: ubuntu-latest
        goarch: arm64
      - os: ubuntu-latest
        goarch: amd64

steps:
  - name: Build
    run: go build .

  - name: Test
    run: go test .
```

## Common Patterns

### Adding a New Platform-Specific Feature

1. **Create implementation file with build tag**
   ```go
   // feature_darwin.go
   //go:build darwin
   ```

2. **Create stub for other platforms**
   ```go
   // feature_linux.go
   //go:build linux

   func NewFeature() (*Feature, error) {
       return nil, fmt.Errorf("feature only available on macOS")
   }
   ```

3. **Write tests with skip logic**
   ```go
   func TestFeature(t *testing.T) {
       f, err := NewFeature()
       if err != nil {
           t.Skipf("Feature not available: %v", err)
       }
       // ... test feature
   }
   ```

### Adding Platform-Specific Tests

1. **Use restrictive build tag**
   ```go
   // graviton_test.go
   //go:build linux && arm64
   ```

2. **Don't reference platform-specific functions in universal tests**
   ```go
   // ❌ BAD - breaks on macOS
   func TestUniversal(t *testing.T) {
       gen := GetGravitonGeneration() // Linux-only function!
   }

   // ✅ GOOD - works everywhere
   func TestUniversal(t *testing.T) {
       result := MatMul(a, b) // Universal function
   }
   ```

## Debugging Build Tag Issues

### Issue: "undefined: SomeFunction"

**Cause**: Calling platform-specific function from universal code

**Fix**: Either:
1. Add build tag to calling file
2. Use runtime detection instead
3. Create stub implementation

### Issue: Test fails on CI but not locally

**Cause**: Test uses platform-specific feature without skip

**Fix**: Add skip check:
```go
backend, err := NewPlatformFeature()
if err != nil {
    t.Skipf("Platform feature not available: %v", err)
}
```

### Issue: "duplicate symbol" or "redeclared"

**Cause**: Two files with incompatible build tags define same symbol

**Fix**: Ensure build tags are mutually exclusive:
```go
// file_darwin.go - //go:build darwin
// file_linux.go  - //go:build linux
```

## Summary

**Build tag strategy**:
- ✅ All platforms can build and test
- ✅ Platform-specific features degrade gracefully
- ✅ Tests skip appropriately on unavailable platforms
- ✅ Clear separation between universal and platform code

**Key insight**: This is a **teaching resource**, so "build everywhere" is more important than "optimize everywhere". Students should learn by running code, not fighting build errors.

## Current Status

- ✅ macOS builds successfully
- ✅ Linux ARM64 builds successfully
- ✅ macOS tests all pass
- ✅ Linux ARM64 tests use proper build tags
- ✅ Platform features degrade gracefully
- ✅ Graviton benchmarks complete on real hardware

**All builds successful!** 🎉
