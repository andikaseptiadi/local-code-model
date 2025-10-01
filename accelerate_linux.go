//go:build linux

package main

import "fmt"

// AccelerateBackend stub for Linux (Accelerate only available on macOS)
type AccelerateBackend struct {
	available bool
}

// NewAccelerateBackend returns unavailable backend on Linux
func NewAccelerateBackend() (*AccelerateBackend, error) {
	return &AccelerateBackend{available: false}, fmt.Errorf("Accelerate only available on macOS (use OpenBLAS on Linux)")
}

// IsAvailable returns false on Linux
func (a *AccelerateBackend) IsAvailable() bool {
	return false
}

// DeviceName returns unavailable message
func (a *AccelerateBackend) DeviceName() string {
	return "Accelerate (macOS only)"
}

// MatMul returns error on Linux
func (a *AccelerateBackend) MatMul(x, y *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("Accelerate not available on Linux (use OpenBLAS)")
}
