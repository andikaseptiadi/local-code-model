//go:build linux

package main

import "fmt"

// MetalBackend stub for Linux (Metal only available on macOS)
type MetalBackend struct {
	available bool
}

// NewMetalBackend returns unavailable backend on Linux
func NewMetalBackend() (*MetalBackend, error) {
	return &MetalBackend{available: false}, fmt.Errorf("Metal only available on macOS")
}

// IsAvailable returns false on Linux
func (m *MetalBackend) IsAvailable() bool {
	return false
}

// DeviceName returns unavailable message
func (m *MetalBackend) DeviceName() string {
	return "Metal (macOS only)"
}

// MatMul returns error on Linux
func (m *MetalBackend) MatMul(a, b *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("Metal not available on Linux")
}

// ANEBackend stub for Linux (ANE only available on macOS)
type ANEBackend struct {
	available bool
}

// NewANEBackend returns unavailable backend on Linux
func NewANEBackend() (*ANEBackend, error) {
	return &ANEBackend{available: false}, fmt.Errorf("ANE only available on macOS")
}

// NewANEBackendWithSize returns unavailable backend on Linux
func NewANEBackendWithSize(m, n, k int) (*ANEBackend, error) {
	return &ANEBackend{available: false}, fmt.Errorf("ANE only available on macOS")
}

// IsAvailable returns false on Linux
func (a *ANEBackend) IsAvailable() bool {
	return false
}

// DeviceName returns unavailable message
func (a *ANEBackend) DeviceName() string {
	return "ANE (macOS only)"
}

// MatMul returns error on Linux
func (a *ANEBackend) MatMul(x, y *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("ANE not available on Linux")
}

// Close does nothing
func (a *ANEBackend) Close() error {
	return nil
}
