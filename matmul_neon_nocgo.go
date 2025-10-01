//go:build linux && arm64 && !cgo

package main

// MatMulSIMD performs matrix multiplication - falls back to naive when CGo disabled
func MatMulSIMD(a, b *Tensor) *Tensor {
	// Without CGo, fall back to simple implementation
	return MatMul(a, b)
}

// IsSIMDAvailable returns false when CGo is disabled
func IsSIMDAvailable() bool {
	return false
}

// SIMDInfo returns information about SIMD availability.
func SIMDInfo() string {
	return "NEON not available (CGo disabled - rebuild with CGO_ENABLED=1)"
}
