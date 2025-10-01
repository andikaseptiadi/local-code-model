//go:build !linux || !arm64

package main

import "fmt"

// SVEBackend stub for non-Linux ARM64 platforms
type SVEBackend struct {
	available bool
}

// NewSVEBackend returns unavailable SVE backend on non-Linux platforms
func NewSVEBackend() (*SVEBackend, error) {
	return &SVEBackend{available: false}, fmt.Errorf("SVE only available on Linux ARM64 (AWS Graviton3+)")
}

// IsAvailable returns false
func (s *SVEBackend) IsAvailable() bool {
	return false
}

// DeviceName returns unavailable message
func (s *SVEBackend) DeviceName() string {
	return "SVE (Linux ARM64 only)"
}

// MatMul returns error
func (s *SVEBackend) MatMul(a, b *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("SVE not available on this platform")
}

// VectorLength returns 0
func (s *SVEBackend) VectorLength() int {
	return 0
}

// Close does nothing
func (s *SVEBackend) Close() error {
	return nil
}
