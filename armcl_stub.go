//go:build !linux || !arm64 || !cgo

package main

import "fmt"

// ARMCLBackend stub for non-Linux ARM64 or non-CGo builds
type ARMCLBackend struct {
	available bool
}

// NewARMCLBackend returns unavailable backend
func NewARMCLBackend() (*ARMCLBackend, error) {
	return &ARMCLBackend{available: false},
		fmt.Errorf("ARM Compute Library only available on Linux ARM64 with CGo")
}

// IsAvailable returns false
func (a *ARMCLBackend) IsAvailable() bool {
	return false
}

// DeviceName returns unavailable message
func (a *ARMCLBackend) DeviceName() string {
	return "ARM Compute Library (Linux ARM64 required)"
}

// MatMul returns error
func (a *ARMCLBackend) MatMul(a1, a2 *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("ARM Compute Library not available on this platform")
}

// GetInfo returns unavailable message
func (a *ARMCLBackend) GetInfo() string {
	return "ARM Compute Library not available (Linux ARM64 + CGo required)"
}

// Close does nothing
func (a *ARMCLBackend) Close() error {
	return nil
}
