package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file defines the abstraction layer for different compute backends:
// CPU, Metal GPU, and Apple Neural Engine. It's about making "stranded
// resources" visible and accessible.
//
// INTENTION:
// Create a unified interface to select hardware acceleration at runtime.
// Show the progression from CPU-only to specialized accelerators. Make it
// explicit what hardware is available and what each is good at.
//
// WHERE THIS SITS ON THE CONTINUUM OF NAIVETE:
//
// This is the INTERFACE layer - it doesn't implement optimizations itself,
// but provides the abstraction to choose between them:
//
// BackendCPU (compute.go, matmul_optimized.go):
//   - Level 0-3: Naive, parallel, cache-blocked
//   - 1-100 GFLOPS depending on optimization level
//   - Always available
//
// BackendMetal (metal.go):
//   - Level 4: GPU acceleration via Metal Performance Shaders
//   - ~4000 GFLOPS (fp32), ~8000 GFLOPS (fp16)
//   - Available on macOS/iOS with Metal-capable GPU
//   - Efficient for matrices >512×512
//   - Overhead: ~1-5ms for data transfer
//
// BackendANE (metal.go):
//   - Level 5: Neural Engine via Core ML
//   - ~38 TOPS (int8), ~19 TFLOPS (fp16)
//   - Available on M-series and A-series chips
//   - Efficient for batched inference with quantized models
//   - Overhead: ~10-50ms for model compilation/loading
//
// THE KEY CONCEPT: STRANDED RESOURCES
// Most code only uses the CPU - even though your system has:
//   - 12-16 CPU cores (M4 Max)
//   - GPU with thousands of shader cores
//   - Neural Engine with 16 cores
//
// Each level of the hierarchy has different characteristics:
//   - CPU: General purpose, low latency, ~100 GFLOPS
//   - GPU: Massively parallel, higher latency, ~4000 GFLOPS
//   - ANE: Specialized for neural nets, highest throughput, ~38 TOPS
//
// PERFORMANCE CHARACTERISTICS:
// Backend selection depends on problem size and precision requirements:
//
// Small problems (n < 256):
//   - CPU wins (overhead of GPU/ANE dominates)
//
// Medium problems (256 < n < 2048):
//   - GPU starts to win for fp16/fp32
//   - CPU still competitive with cache blocking
//
// Large problems (n > 2048):
//   - GPU strongly preferred for training
//   - ANE preferred for int8 inference
//
// WHY THIS APPROACH:
// By exposing the backend as a configuration choice, we:
//   1. Make performance tradeoffs visible and controllable
//   2. Allow users to benchmark their specific workloads
//   3. Support development (CPU) and production (GPU/ANE) from same code
//   4. Teach how modern ML systems actually work under the hood
//
// ===========================================================================
// RECOMMENDED READING:
//
// Metal Programming Guide:
// - "Metal Programming Guide" by Apple
//   https://developer.apple.com/metal/
//
// - "Metal Performance Shaders" documentation
//   https://developer.apple.com/documentation/metalperformanceshaders
//
// Neural Engine:
// - "Core ML Performance" by Apple
//   https://developer.apple.com/documentation/coreml/core_ml_api/optimizing_performance
//
// - "Integrating a Core ML Model into Your App"
//   Shows ANE integration patterns
// ===========================================================================

// ComputeBackend represents different hardware accelerators available on the system.
type ComputeBackend int

const (
	// BackendCPU uses standard Go CPU execution (multi-threaded).
	BackendCPU ComputeBackend = iota

	// BackendMetal uses Metal Performance Shaders for GPU acceleration.
	// Available on macOS/iOS with Metal-capable GPUs.
	// Best for: large batches, fp16 operations, >512×512 matrices.
	BackendMetal

	// BackendANE uses Apple Neural Engine via Core ML.
	// Available on M-series and A-series chips.
	// Best for: inference, int8/fp16 quantized models, batch operations.
	BackendANE

	// BackendAuto selects the best available backend based on operation size
	// and available hardware.
	BackendAuto
)

// BackendCapabilities describes what a backend can efficiently handle.
type BackendCapabilities struct {
	// Available indicates if this backend is present on the system.
	Available bool

	// Name is the human-readable backend name.
	Name string

	// MinEfficientSize is the problem size where this backend becomes efficient.
	// Below this size, overhead dominates performance.
	MinEfficientSize int

	// MaxBatchSize is the maximum batch size this backend can handle efficiently.
	// Zero means unlimited.
	MaxBatchSize int

	// SupportsFloat16 indicates if fp16 operations are supported.
	SupportsFloat16 bool

	// SupportsInt8 indicates if int8 quantized operations are supported.
	SupportsInt8 bool

	// PeakGFLOPS is the theoretical peak performance in GFLOPS.
	PeakGFLOPS float64

	// MemoryBandwidthGBps is the memory bandwidth in GB/s.
	MemoryBandwidthGBps float64
}

// BackendConfig extends ComputeConfig with backend selection.
type BackendConfig struct {
	ComputeConfig

	// Backend specifies which compute backend to use.
	Backend ComputeBackend

	// PreferANE indicates preference for ANE over Metal when both available.
	// ANE is more power-efficient but has more restrictions.
	PreferANE bool

	// AllowFallback allows falling back to CPU if accelerator unavailable.
	AllowFallback bool
}

// DefaultBackendConfig returns a configuration that auto-selects the best backend.
func DefaultBackendConfig() BackendConfig {
	return BackendConfig{
		ComputeConfig: DefaultComputeConfig(),
		Backend:       BackendAuto,
		PreferANE:     true,  // Power efficiency
		AllowFallback: true,  // Robustness
	}
}

// CPUOnlyConfig returns a configuration that only uses CPU.
func CPUOnlyConfig() BackendConfig {
	return BackendConfig{
		ComputeConfig: DefaultComputeConfig(),
		Backend:       BackendCPU,
		PreferANE:     false,
		AllowFallback: false,
	}
}

// MetalConfig returns a configuration optimized for Metal GPU.
func MetalConfig() BackendConfig {
	return BackendConfig{
		ComputeConfig: ComputeConfig{
			Parallel:           true,
			NumWorkers:         1, // Metal handles parallelism internally
			MinSizeForParallel: 128,
		},
		Backend:       BackendMetal,
		PreferANE:     false,
		AllowFallback: true,
	}
}

// ANEConfig returns a configuration optimized for Neural Engine.
func ANEConfig() BackendConfig {
	return BackendConfig{
		ComputeConfig: ComputeConfig{
			Parallel:           false, // ANE is single-threaded from Go perspective
			NumWorkers:         1,
			MinSizeForParallel: 0,
		},
		Backend:       BackendANE,
		PreferANE:     true,
		AllowFallback: true,
	}
}

// BackendInfo provides runtime information about available backends.
type BackendInfo struct {
	CPU   BackendCapabilities
	Metal BackendCapabilities
	ANE   BackendCapabilities
}

// DetectBackends probes the system for available compute backends.
func DetectBackends() BackendInfo {
	info := BackendInfo{
		CPU: BackendCapabilities{
			Available:           true,
			Name:                "CPU (Go Runtime)",
			MinEfficientSize:    64,
			MaxBatchSize:        0, // Unlimited
			SupportsFloat16:     false,
			SupportsInt8:        false,
			PeakGFLOPS:          estimateCPUGFLOPS(),
			MemoryBandwidthGBps: estimateCPUBandwidth(),
		},
	}

	// Check for Metal availability
	if metalAvailable() {
		info.Metal = BackendCapabilities{
			Available:           true,
			Name:                "Metal Performance Shaders",
			MinEfficientSize:    256,  // GPU overhead is significant
			MaxBatchSize:        1024, // Limited by GPU memory
			SupportsFloat16:     true,
			SupportsInt8:        true,
			PeakGFLOPS:          estimateMetalGFLOPS(),
			MemoryBandwidthGBps: estimateMetalBandwidth(),
		}
	}

	// Check for ANE availability
	if aneAvailable() {
		info.ANE = BackendCapabilities{
			Available:           true,
			Name:                "Apple Neural Engine",
			MinEfficientSize:    512,   // ANE has high dispatch overhead
			MaxBatchSize:        128,   // ANE prefers smaller batches
			SupportsFloat16:     true,
			SupportsInt8:        true,
			PeakGFLOPS:          38000, // 38 TOPS for M4
			MemoryBandwidthGBps: 400,   // Shared with CPU
		}
	}

	return info
}

// SelectBackend chooses the optimal backend for a given operation size.
func (info BackendInfo) SelectBackend(cfg BackendConfig, opSize int) ComputeBackend {
	if cfg.Backend != BackendAuto {
		// Explicit backend selection
		return cfg.Backend
	}

	// Auto-selection logic based on operation size and capabilities

	// ANE is best for large batched operations at inference time
	if cfg.PreferANE && info.ANE.Available && opSize >= info.ANE.MinEfficientSize {
		return BackendANE
	}

	// Metal is good for medium-to-large operations
	if info.Metal.Available && opSize >= info.Metal.MinEfficientSize {
		return BackendMetal
	}

	// Fall back to CPU for everything else
	return BackendCPU
}

// Placeholder functions for backend detection
// These will be implemented with CGo/syscalls

func metalAvailable() bool {
	// Check if running on macOS and Metal is available
	// This would use CGo to call Metal availability check
	return false // Placeholder - implement with CGo
}

func aneAvailable() bool {
	// Check if running on M-series or A-series with ANE
	// This would check Core ML availability
	return false // Placeholder - implement with CGo
}

func estimateCPUGFLOPS() float64 {
	// Rough estimate based on CPU benchmarks
	// M4 Max: ~12 cores * ~3.5 GHz * 32 FLOPS/cycle ≈ 1344 GFLOPS (theoretical)
	// Actual sustained: ~100-200 GFLOPS in practice
	return 150.0
}

func estimateCPUBandwidth() float64 {
	// M4 Max unified memory bandwidth
	return 400.0 // GB/s
}

func estimateMetalGFLOPS() float64 {
	// M4 Max GPU: ~4 TFLOPS (fp32), ~8 TFLOPS (fp16)
	return 4000.0
}

func estimateMetalBandwidth() float64 {
	// Shares unified memory with CPU
	return 400.0 // GB/s
}

// ResourceUtilization tracks which resources are being used vs stranded.
type ResourceUtilization struct {
	// CPU cores utilized (0.0 - 1.0 per core)
	CPUUtilization []float64

	// GPU utilization (0.0 - 1.0)
	GPUUtilization float64

	// ANE utilization (0.0 - 1.0)
	ANEUtilization float64

	// Memory bandwidth utilization (0.0 - 1.0)
	MemoryBandwidthUtilization float64

	// StrandedResources lists resources not being utilized
	StrandedResources []string
}

// EstimateUtilization estimates resource utilization for a given backend and operation.
func EstimateUtilization(backend ComputeBackend, opSize int) ResourceUtilization {
	util := ResourceUtilization{
		CPUUtilization: make([]float64, 0),
	}

	info := DetectBackends()

	switch backend {
	case BackendCPU:
		// Using CPU cores, GPU/ANE stranded
		numWorkers := globalComputeConfig.numWorkers()
		util.CPUUtilization = make([]float64, numWorkers)
		for i := range util.CPUUtilization {
			// Naive estimate: assume full utilization during compute
			util.CPUUtilization[i] = 0.8 // 80% due to memory stalls
		}
		util.MemoryBandwidthUtilization = 0.3 // Typically only 30% utilized

		if info.Metal.Available {
			util.StrandedResources = append(util.StrandedResources, "GPU (Metal)")
		}
		if info.ANE.Available {
			util.StrandedResources = append(util.StrandedResources, "Neural Engine")
		}

	case BackendMetal:
		// Using GPU, ANE stranded, minimal CPU
		util.GPUUtilization = 0.9
		util.MemoryBandwidthUtilization = 0.7 // Better utilization than CPU
		util.CPUUtilization = []float64{0.05} // Just for dispatch

		if info.ANE.Available {
			util.StrandedResources = append(util.StrandedResources, "Neural Engine")
		}

	case BackendANE:
		// Using ANE, GPU stranded, minimal CPU
		util.ANEUtilization = 0.95
		util.MemoryBandwidthUtilization = 0.5
		util.CPUUtilization = []float64{0.02} // Minimal dispatch overhead

		if info.Metal.Available {
			util.StrandedResources = append(util.StrandedResources, "GPU (Metal)")
		}
	}

	return util
}

// PrintBackendInfo displays available backends and their capabilities.
func PrintBackendInfo() {
	info := DetectBackends()

	println("=== Available Compute Backends ===\n")

	// CPU
	println("CPU:")
	println("  Available:   ", info.CPU.Available)
	println("  Peak GFLOPS: ", info.CPU.PeakGFLOPS)
	println("  Bandwidth:   ", info.CPU.MemoryBandwidthGBps, "GB/s")
	println("  Min Size:    ", info.CPU.MinEfficientSize)
	println()

	// Metal
	println("Metal GPU:")
	println("  Available:   ", info.Metal.Available)
	if info.Metal.Available {
		println("  Peak GFLOPS: ", info.Metal.PeakGFLOPS)
		println("  Bandwidth:   ", info.Metal.MemoryBandwidthGBps, "GB/s")
		println("  Min Size:    ", info.Metal.MinEfficientSize)
		println("  FP16:        ", info.Metal.SupportsFloat16)
	}
	println()

	// ANE
	println("Neural Engine:")
	println("  Available:   ", info.ANE.Available)
	if info.ANE.Available {
		println("  Peak TOPS:   ", info.ANE.PeakGFLOPS/1000.0)
		println("  Bandwidth:   ", info.ANE.MemoryBandwidthGBps, "GB/s")
		println("  Min Size:    ", info.ANE.MinEfficientSize)
		println("  FP16/INT8:   ", info.ANE.SupportsFloat16, "/", info.ANE.SupportsInt8)
	}
	println()
}

// Global backend info (initialized once)
var globalBackendInfo BackendInfo

func init() {
	globalBackendInfo = DetectBackends()
}