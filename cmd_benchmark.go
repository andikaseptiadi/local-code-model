package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file implements a command-line benchmark runner that can test
// performance across diverse ARM architectures and hardware configurations.
//
// INTENTION:
// Create a comprehensive benchmarking tool that works on:
//   - Apple Silicon (M1, M2, M3, M4 - all variants)
//   - AWS Graviton2 (ARM Neoverse N1)
//   - AWS Graviton3 (ARM Neoverse V1)
//   - AWS Graviton3E (Neoverse V1 + bfloat16)
//   - AWS Graviton4 (ARM Neoverse V2)
//   - AWS g5g (Graviton2 + NVIDIA T4G GPU)
//   - Other ARM64 platforms
//
// WHY THIS MATTERS:
// Each architecture has unique characteristics that affect performance:
//
// Graviton2 (Neoverse N1):
//   - 64 KB L1, 1 MB L2 per core
//   - DDR4 memory (~150 GB/s)
//   - ARM v8.2 + NEON
//   - Baseline for Graviton family
//
// Graviton3 (Neoverse V1):
//   - 64 KB L1, 1 MB L2, 32-64 MB L3
//   - DDR5 memory (~307 GB/s)
//   - ARM v8.4 + SVE 256-bit
//   - 2x memory bandwidth vs Graviton2
//
// Graviton3E:
//   - Same as Graviton3
//   - + bfloat16 support (for ML)
//   - Important for ML inference workloads
//
// Graviton4 (Neoverse V2):
//   - 64 KB L1, 2 MB L2, 64 MB L3 (larger caches!)
//   - DDR5 memory (~500 GB/s)
//   - ARM v9.0 + SVE2 256-bit
//   - 60% faster than Graviton3
//
// g5g (Graviton2 + NVIDIA T4G):
//   - Graviton2 CPU specs
//   - + NVIDIA T4G GPU (16 GB, ~65 TFLOPS fp16)
//   - Shows CPU+GPU hybrid performance
//
// M4 Max (Apple Silicon):
//   - 192 KB L1 per P-core, 16 MB L2
//   - Unified memory (~400 GB/s)
//   - + GPU (40 cores, ~4 TFLOPS)
//   - + Neural Engine (38 TOPS)
//   - Largest caches, most specialized hardware
//
// EXPECTED PERFORMANCE PATTERNS:
//
// Naive (single-threaded):
//   - All architectures: 1-3 GFLOPS (similar, limited by single core)
//
// Parallel:
//   - Graviton2: 2-5x speedup (memory bandwidth limited)
//   - Graviton3: 3-8x speedup (better bandwidth)
//   - Graviton4: 4-10x speedup (best bandwidth)
//   - M4 Max: 2-5x speedup (unified memory helps, but still bandwidth bound)
//
// Cache-Blocked:
//   - Graviton2/3: 3-5x improvement (smaller caches)
//   - Graviton4: 4-8x improvement (larger L2/L3)
//   - M4 Max: 5-10x improvement (largest caches)
//
// Cache-Blocked + Parallel:
//   - Graviton2: 10-20x total
//   - Graviton3: 15-30x total
//   - Graviton4: 20-40x total
//   - M4 Max: 20-50x total
//
// GPU (when available):
//   - g5g + T4G: 100-500x (GPU optimized for this)
//   - M4 Max + Metal: 50-200x (Metal overhead higher)
//
// ===========================================================================

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
)

// BenchmarkConfig holds command-line options.
type BenchmarkConfig struct {
	Sizes          []int
	Iterations     int
	OutputJSON     string
	OutputCSV      string
	Visualize      bool
	VisualizeFmt   string
	DetectHW       bool
	CompareFiles   []string
	QuickMode      bool
	IncludeGPU     bool
}

// RunBenchmarkCommand is the main entry point for the benchmark command.
func RunBenchmarkCommand(args []string) error {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)

	config := BenchmarkConfig{}

	var sizesStr string
	fs.StringVar(&sizesStr, "sizes", "64,128,256,512", "Matrix sizes to benchmark (comma-separated)")
	fs.IntVar(&config.Iterations, "iterations", 5, "Number of iterations per benchmark")
	fs.StringVar(&config.OutputJSON, "json", "", "Output JSON file")
	fs.StringVar(&config.OutputCSV, "csv", "", "Output CSV file")
	fs.BoolVar(&config.Visualize, "visualize", false, "Generate visualizations")
	fs.StringVar(&config.VisualizeFmt, "format", "ascii", "Visualization format (ascii, gnuplot, csv)")
	fs.BoolVar(&config.DetectHW, "detect", false, "Only detect hardware and exit")
	fs.BoolVar(&config.QuickMode, "quick", false, "Quick mode (fewer sizes and iterations)")
	fs.BoolVar(&config.IncludeGPU, "gpu", false, "Include GPU benchmarks if available")

	fs.Parse(args)

	// Parse sizes
	for _, s := range strings.Split(sizesStr, ",") {
		var size int
		fmt.Sscanf(s, "%d", &size)
		if size > 0 {
			config.Sizes = append(config.Sizes, size)
		}
	}

	// Quick mode overrides
	if config.QuickMode {
		config.Sizes = []int{128, 256}
		config.Iterations = 3
	}

	// Default sizes if none specified
	if len(config.Sizes) == 0 {
		config.Sizes = []int{64, 128, 256, 512}
	}

	// Hardware detection only
	if config.DetectHW {
		return runHardwareDetection()
	}

	// Run benchmarks
	return runBenchmarks(config)
}

// runHardwareDetection detects and prints hardware information.
func runHardwareDetection() error {
	fmt.Println("=== Hardware Detection ===")
	fmt.Println()

	hw := DetectHardware()

	fmt.Printf("Operating System: %s\n", hw.OS)
	fmt.Printf("Architecture:     %s\n", hw.Arch)
	fmt.Printf("CPU Model:        %s\n", hw.CPUModel)
	fmt.Printf("CPU Cores:        %d\n", hw.NumCPU)
	fmt.Println()

	fmt.Printf("Cache Hierarchy:\n")
	if hw.L1CacheKB > 0 {
		fmt.Printf("  L1: %d KB\n", hw.L1CacheKB)
	}
	if hw.L2CacheKB > 0 {
		fmt.Printf("  L2: %d KB\n", hw.L2CacheKB)
	}
	if hw.L3CacheKB > 0 {
		fmt.Printf("  L3: %d KB\n", hw.L3CacheKB)
	}
	fmt.Println()

	fmt.Printf("Features:\n")
	if hw.HasNEON {
		fmt.Printf("  ✓ ARM NEON (128-bit SIMD)\n")
	}
	if hw.HasSVE {
		fmt.Printf("  ✓ ARM SVE (Scalable Vector Extension)\n")
	}
	if hw.HasMetal {
		fmt.Printf("  ✓ Metal GPU acceleration available\n")
	}
	if hw.HasANE {
		fmt.Printf("  ✓ Apple Neural Engine available\n")
	}
	fmt.Println()

	// Detect specific platform
	platform := detectPlatform(hw)
	fmt.Printf("Detected Platform: %s\n", platform)
	fmt.Println()

	printPlatformCharacteristics(platform)

	return nil
}

// detectPlatform identifies the specific ARM platform.
func detectPlatform(hw HardwareInfo) string {
	if hw.OS == "darwin" && hw.Arch == "arm64" {
		// Apple Silicon
		if hw.HasANE {
			return "Apple Silicon (M-series)"
		}
		return "Apple Silicon (A-series)"
	}

	if hw.OS == "linux" && hw.Arch == "arm64" {
		// Check for AWS Graviton indicators
		// In production, would check:
		// - /sys/devices/system/cpu/cpu0/regs/identification/midr_el1
		// - /proc/cpuinfo model name
		// - DMI product name

		if hw.HasSVE {
			// Graviton3/3E/4 all have SVE
			// Would need more sophisticated detection for exact model
			return "AWS Graviton3/3E/4 (Neoverse V1/V2)"
		}

		// Graviton2 has NEON but not SVE
		return "AWS Graviton2 (Neoverse N1)"
	}

	return "Unknown ARM64 platform"
}

// printPlatformCharacteristics shows expected performance characteristics.
func printPlatformCharacteristics(platform string) {
	fmt.Println("Expected Characteristics:")
	fmt.Println()

	switch {
	case strings.Contains(platform, "M-series"):
		fmt.Println("Apple M-series (M1/M2/M3/M4):")
		fmt.Println("  • Large L1/L2 caches (excellent cache blocking)")
		fmt.Println("  • Unified memory architecture")
		fmt.Println("  • High memory bandwidth (200-400 GB/s)")
		fmt.Println("  • Metal GPU acceleration available")
		fmt.Println("  • Neural Engine for ML inference")
		fmt.Println("  • Best for: Development, on-device inference")

	case strings.Contains(platform, "Graviton2"):
		fmt.Println("AWS Graviton2 (ARM Neoverse N1):")
		fmt.Println("  • Small-medium caches (64KB L1, 1MB L2)")
		fmt.Println("  • DDR4 memory (~150 GB/s)")
		fmt.Println("  • ARM v8.2 + NEON only")
		fmt.Println("  • Cost-effective baseline")
		fmt.Println("  • Best for: General purpose, cost optimization")
		fmt.Println("  • Note: Check for g5g (adds NVIDIA T4G GPU)")

	case strings.Contains(platform, "Graviton3"):
		fmt.Println("AWS Graviton3/3E/4 (ARM Neoverse V1/V2):")
		fmt.Println("  • Larger caches (64KB L1, 1-2MB L2, 32-64MB L3)")
		fmt.Println("  • DDR5 memory (307-500 GB/s)")
		fmt.Println("  • ARM v8.4/v9.0 + SVE/SVE2 256-bit")
		fmt.Println("  • Graviton3E adds bfloat16 for ML")
		fmt.Println("  • Graviton4: 60% faster than Graviton3")
		fmt.Println("  • Best for: Compute-intensive, HPC, ML inference")

	default:
		fmt.Println("Unknown platform - will measure and characterize")
	}

	fmt.Println()
}

// runBenchmarks executes the benchmark suite.
func runBenchmarks(config BenchmarkConfig) error {
	fmt.Println("=== Starting Benchmark Suite ===")
	fmt.Println()

	// Run hardware detection first
	hw := DetectHardware()
	platform := detectPlatform(hw)

	fmt.Printf("Platform: %s\n", platform)
	fmt.Printf("Cores:    %d\n", hw.NumCPU)
	fmt.Printf("Sizes:    %v\n", config.Sizes)
	fmt.Printf("Iterations: %d\n", config.Iterations)
	fmt.Println()

	// Run benchmark suite
	suite := RunBenchmarkSuite(config.Sizes, config.Iterations)

	// Print summary
	suite.PrintSummary()

	// Save JSON if requested
	if config.OutputJSON != "" {
		if err := saveBenchmarkJSON(suite, config.OutputJSON); err != nil {
			return fmt.Errorf("failed to save JSON: %w", err)
		}
		fmt.Printf("Saved results to %s\n", config.OutputJSON)
	}

	// Generate CSV if requested
	if config.OutputCSV != "" {
		if err := saveBenchmarkCSV(suite, config.OutputCSV); err != nil {
			return fmt.Errorf("failed to save CSV: %w", err)
		}
		fmt.Printf("Saved results to %s\n", config.OutputCSV)
	}

	// Visualize if requested
	if config.Visualize {
		vizConfig := DefaultVisualizationConfig()
		vizConfig.Format = config.VisualizeFmt

		if err := GenerateVisualization(suite, vizConfig); err != nil {
			return fmt.Errorf("failed to generate visualization: %w", err)
		}
	}

	// Print collection instructions
	printCollectionInstructions(platform, config.OutputJSON)

	return nil
}

// saveBenchmarkJSON saves benchmark results to a JSON file.
func saveBenchmarkJSON(suite *BenchmarkSuite, filename string) error {
	data, err := json.MarshalIndent(suite, "", "  ")
	if err != nil {
		return err
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.Write(data)
	return err
}

// saveBenchmarkCSV saves benchmark results to a CSV file.
func saveBenchmarkCSV(suite *BenchmarkSuite, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Redirect stdout to file temporarily
	oldStdout := os.Stdout
	os.Stdout = f

	generateCSV(suite, DefaultVisualizationConfig())

	os.Stdout = oldStdout

	return nil
}

// printCollectionInstructions shows how to collect benchmarks across architectures.
func printCollectionInstructions(platform, jsonFile string) {
	fmt.Println()
	fmt.Println("=== Cross-Architecture Comparison ===")
	fmt.Println()
	fmt.Println("To compare performance across different ARM platforms:")
	fmt.Println()
	fmt.Println("1. Run benchmarks on each platform:")
	fmt.Println()
	fmt.Println("   # On M4 Max (macOS)")
	fmt.Println("   go run . benchmark -json=m4_max.json")
	fmt.Println()
	fmt.Println("   # On Graviton2 (AWS c6g)")
	fmt.Println("   go run . benchmark -json=graviton2.json")
	fmt.Println()
	fmt.Println("   # On Graviton3 (AWS c7g)")
	fmt.Println("   go run . benchmark -json=graviton3.json")
	fmt.Println()
	fmt.Println("   # On Graviton3E (AWS c7gn with enhanced networking)")
	fmt.Println("   go run . benchmark -json=graviton3e.json")
	fmt.Println()
	fmt.Println("   # On Graviton4 (AWS c8g)")
	fmt.Println("   go run . benchmark -json=graviton4.json")
	fmt.Println()
	fmt.Println("   # On g5g (Graviton2 + NVIDIA T4G)")
	fmt.Println("   go run . benchmark -json=g5g.json -gpu")
	fmt.Println()
	fmt.Println("2. Collect all JSON files to one machine")
	fmt.Println()
	fmt.Println("3. Generate comparison visualizations:")
	fmt.Println("   go run . compare -files=\"*.json\" -format=gnuplot")
	fmt.Println()
	fmt.Println("This will reveal:")
	fmt.Println("  • How cache sizes affect cache-blocking gains")
	fmt.Println("  • How memory bandwidth affects parallel scaling")
	fmt.Println("  • How SVE vs NEON affects vectorization")
	fmt.Println("  • Where each architecture excels")
	fmt.Println()

	if jsonFile != "" {
		fmt.Printf("Current benchmark saved to: %s\n", jsonFile)
		fmt.Printf("Platform: %s\n", platform)
	}

	fmt.Println()
}

// printASCIILogo prints a fun ASCII logo.
func printASCIILogo() {
	logo := `
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   Matrix Multiplication Performance Atlas                ║
║   Mapping the Optimization Continuum                     ║
║                                                           ║
║   From 1 GFLOPS to 38,000 GFLOPS                        ║
║   Across ARM architectures                               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
`
	fmt.Println(logo)
}

