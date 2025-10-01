//go:build !linux || !cgo

package main

import "fmt"

// OpenBLASBackend stub for non-Linux or non-CGo builds
type OpenBLASBackend struct {
	available bool
}

// NewOpenBLASBackend returns unavailable backend on non-Linux
func NewOpenBLASBackend() (*OpenBLASBackend, error) {
	return &OpenBLASBackend{available: false},
		fmt.Errorf("OpenBLAS only available on Linux with CGo enabled")
}

// IsAvailable returns false
func (o *OpenBLASBackend) IsAvailable() bool {
	return false
}

// DeviceName returns unavailable message
func (o *OpenBLASBackend) DeviceName() string {
	return "OpenBLAS (Linux only)"
}

// MatMul returns error
func (o *OpenBLASBackend) MatMul(a, b *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("OpenBLAS not available on this platform")
}

// Close does nothing
func (o *OpenBLASBackend) Close() error {
	return nil
}

// GetInfo returns unavailable message
func (o *OpenBLASBackend) GetInfo() string {
	return "OpenBLAS not available (Linux + CGo required)"
}
