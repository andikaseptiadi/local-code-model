package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file defines a minimal backend configuration for pure Go transformer
// training and inference. This is intentionally simple - just CPU execution
// with configurable parallelism.
//
// INTENTION:
// Provide a clean, portable implementation that runs anywhere Go runs.
// No CGO dependencies, no platform-specific code, no external libraries.
// Just pure, idiomatic Go that demonstrates transformer implementation.
//
// WHERE THIS SITS:
// This is the TRAINING/INFERENCE layer - focused on correctness and clarity
// rather than maximum performance. The optimization journey (cache blocking,
// SIMD, GPU) is interesting for learning, but orthogonal to understanding
// transformer architecture and training dynamics.
//
// For production inference at scale, you'd want:
//   - Quantization (int8/int4) for memory efficiency
//   - Batch processing for throughput
//   - GPU acceleration (CUDA/Metal) for large models
//   - Model serving infrastructure (REST API, load balancing)
//
// But for learning, experimentation, and small models, pure Go is perfect:
//   - No build complexity
//   - Works on any platform
//   - Easy to debug and modify
//   - Fast enough for models <100M parameters
//
// ===========================================================================

// BackendConfig provides minimal configuration for tensor operations.
// For training, we use single-threaded execution for determinism.
type BackendConfig struct {
	ComputeConfig
}

// DefaultBackendConfig returns the default configuration for training/inference.
func DefaultBackendConfig() BackendConfig {
	return BackendConfig{
		ComputeConfig: DefaultComputeConfig(),
	}
}

// CPUOnlyConfig returns a CPU-only configuration (same as default).
func CPUOnlyConfig() BackendConfig {
	return DefaultBackendConfig()
}
