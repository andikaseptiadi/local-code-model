package main

import "runtime"

// ===========================================================================
// COMPUTE CONFIGURATION - Minimal stub for training code
// ===========================================================================
//
// This file provides the minimal ComputeConfig infrastructure needed by
// the tensor and backend code. For training, we use single-threaded execution
// for determinism and simplicity.
//
// The full parallel implementation with goroutines and cache-blocked matmul
// lives in ../arm-benchmark/compute.go for benchmarking purposes.
//
// ===========================================================================

// ComputeConfig controls parallelization behavior for tensor operations.
type ComputeConfig struct {
	// Parallel enables multi-threaded execution of tensor operations.
	Parallel bool

	// NumWorkers specifies the number of worker goroutines to use.
	// If 0, defaults to runtime.NumCPU().
	NumWorkers int

	// MinSizeForParallel specifies the minimum matrix dimension
	// before parallelization is used.
	MinSizeForParallel int
}

// DefaultComputeConfig returns a single-threaded configuration for training.
//
// Why single-threaded? Training with backpropagation requires careful
// management of computational graphs and gradient flows. Single-threaded
// execution ensures:
//   1. Deterministic results (reproducible training runs)
//   2. Easier debugging (no race conditions)
//   3. Simpler implementation (no synchronization overhead)
//
// For inference at scale, use parallel execution via ParallelMatMul.
func DefaultComputeConfig() ComputeConfig {
	return ComputeConfig{
		Parallel:           false, // Single-threaded for determinism
		NumWorkers:         1,
		MinSizeForParallel: 0,
	}
}

// SingleThreadedConfig returns a configuration for single-threaded execution.
func SingleThreadedConfig() ComputeConfig {
	return ComputeConfig{
		Parallel:           false,
		NumWorkers:         1,
		MinSizeForParallel: 0,
	}
}

// numWorkers returns the actual number of workers to use.
func (c ComputeConfig) numWorkers() int {
	if !c.Parallel {
		return 1
	}
	if c.NumWorkers > 0 {
		return c.NumWorkers
	}
	return runtime.NumCPU()
}

// shouldParallelize determines if an operation should use parallelization.
func (c ComputeConfig) shouldParallelize(size int) bool {
	return c.Parallel && size >= c.MinSizeForParallel
}

// Global compute configuration
var globalComputeConfig = DefaultComputeConfig()

// SetGlobalComputeConfig sets the global compute configuration.
func SetGlobalComputeConfig(cfg ComputeConfig) {
	globalComputeConfig = cfg
}

// GetGlobalComputeConfig returns the current global compute configuration.
func GetGlobalComputeConfig() ComputeConfig {
	return globalComputeConfig
}

// MatMulWithConfig performs matrix multiplication with the specified config.
//
// For training, this always uses the standard single-threaded MatMul
// from tensor.go to ensure deterministic gradient computation.
func MatMulWithConfig(a, b *Tensor, cfg ComputeConfig) *Tensor {
	// For now, always use standard MatMul regardless of config
	// This ensures training is deterministic
	return MatMul(a, b)
}
