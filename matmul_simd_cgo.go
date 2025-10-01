//go:build darwin && arm64 && cgo

package main

// ===========================================================================
// SIMD Fallback for CGo Builds
// ===========================================================================
//
// Go doesn't allow mixing CGo and assembly in the same package. When building
// with CGo (for Metal/Accelerate), we fall back to cache-blocked
// implementation for SIMD strategy.
//
// WHY THIS LIMITATION EXISTS:
//
// CGo compilation involves:
//   1. Go compiler processes .go files
//   2. C/Objective-C compiler processes .c/.m files
//   3. Complex linker combines both
//
// Assembly (.s) files are processed by the Go assembler, which can conflict
// with CGo's linker expectations. The Go toolchain doesn't support both in
// the same package.
//
// PRACTICAL IMPACT:
//
// When building with tags "darwin,cgo" (for Metal/Accelerate):
//   - This file is used (fallback to cache-blocked)
//   - matmul_simd.go is NOT compiled (excluded by !cgo tag)
//   - matmul_neon_arm64.s is NOT used (Go prevents it)
//
// When building without CGo (CPU-only):
//   - matmul_simd.go is used (ARM64 NEON assembly)
//   - This file is NOT compiled (excluded by cgo tag)
//
// NOT A PROBLEM IN PRACTICE:
//
// Accelerate (~200 GFLOPS) is already much faster than SIMD (~20-40 GFLOPS),
// so falling back to cache-blocked when CGo is enabled is fine.
//
// The SIMD implementation still serves its educational purpose:
//   - Shows instruction-level parallelism
//   - Demonstrates NEON vectorization
//   - Can be benchmarked in CPU-only builds
//
// BUILD INSTRUCTIONS:
//
// With Metal/Accelerate (this file used):
//   go build -tags darwin,cgo .
//
// CPU-only with SIMD (matmul_simd.go used):
//   go build .  # without CGo tags
//
// ===========================================================================

// MatMulSIMD falls back to cache-blocked when CGo is enabled.
//
// This is necessary because Go doesn't allow mixing CGo and assembly in the
// same package. Since Metal and Accelerate require CGo, SIMD assembly is
// not available in CGo builds.
//
// Recommendation: Use StrategyAccelerate instead, which is typically faster.
func MatMulSIMD(a, b *Tensor) *Tensor {
	// Fall back to cache-blocked implementation
	// Note: If you need SIMD assembly, build without -tags darwin,cgo
	return MatMulCacheBlocked(a, b, 64)
}

// IsSIMDAvailable returns false when CGo is enabled.
func IsSIMDAvailable() bool {
	return false // Assembly SIMD not available with CGo
}

// SIMDInfo returns information about SIMD availability.
func SIMDInfo() string {
	return "SIMD (assembly) not available in CGo builds. Use StrategyAccelerate instead (faster anyway)."
}
