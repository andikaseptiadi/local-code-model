// +build !arm64

package main

// ===========================================================================
// SIMD Stub for Non-ARM64 Platforms
// ===========================================================================
//
// This file provides stub implementations for platforms that don't support
// ARM NEON instructions.
//
// On x86_64, you could implement AVX/AVX2/AVX-512 versions.
// On other architectures, fall back to cache-blocked implementation.
//
// ===========================================================================

// MatMulSIMD falls back to cache-blocked implementation on non-ARM64 platforms.
func MatMulSIMD(a, b *Tensor) *Tensor {
	// Fall back to cache-blocked implementation
	return MatMulCacheBlocked(a, b, 64)
}

// IsSIMDAvailable returns false on non-ARM64 platforms (for now).
func IsSIMDAvailable() bool {
	return false
}

// SIMDInfo returns information about SIMD availability.
func SIMDInfo() string {
	return "SIMD not implemented for this architecture (ARM64 only)"
}
