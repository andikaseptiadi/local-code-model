package main

import (
	"fmt"
	"testing"
)

// Comprehensive benchmark comparing all backends at various sizes
func BenchmarkComprehensive(b *testing.B) {
	sizes := []int{256, 512, 1024, 2048}

	for _, size := range sizes {
		// ANE/MPSGraph
		b.Run(fmt.Sprintf("ANE_%d", size), func(b *testing.B) {
			backend, err := NewANEBackendWithSize(size, size, size)
			if err != nil {
				b.Skipf("ANE not available: %v", err)
			}

			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}
		})

		// Metal
		b.Run(fmt.Sprintf("Metal_%d", size), func(b *testing.B) {
			backend, err := NewMetalBackend()
			if err != nil {
				b.Skipf("Metal not available: %v", err)
			}

			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}
		})

		// Accelerate
		b.Run(fmt.Sprintf("Accelerate_%d", size), func(b *testing.B) {
			backend, err := NewAccelerateBackend()
			if err != nil {
				b.Skipf("Accelerate not available: %v", err)
			}

			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}
		})
	}
}
