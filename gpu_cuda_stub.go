//go:build !linux || !cgo

package main

import "fmt"

// CUDABackend stub for non-Linux or non-CGo builds
type CUDABackend struct {
	available bool
}

// DeviceProperties stub
type DeviceProperties struct {
	Name                string
	TotalGlobalMem      uint64
	ComputeCapability   string
	MultiProcessorCount int
	ClockRate           int
	SharedMemPerBlock   uint64
}

// NewCUDABackend returns unavailable backend on non-Linux
func NewCUDABackend() (*CUDABackend, error) {
	return &CUDABackend{available: false},
		fmt.Errorf("CUDA only available on Linux with CGo enabled and CUDA installed")
}

// IsAvailable returns false
func (c *CUDABackend) IsAvailable() bool {
	return false
}

// DeviceName returns unavailable message
func (c *CUDABackend) DeviceName() string {
	return "CUDA (Linux + NVIDIA GPU required)"
}

// MatMul returns error
func (c *CUDABackend) MatMul(a, b *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("CUDA not available on this platform")
}

// GetInfo returns unavailable message
func (c *CUDABackend) GetInfo() string {
	return "CUDA not available (Linux + CGo + CUDA runtime required)"
}

// Close does nothing
func (c *CUDABackend) Close() error {
	return nil
}
