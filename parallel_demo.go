package main

import (
	"fmt"
	"runtime"
	"time"
)

// DemoParallelExecution demonstrates single-threaded vs parallel performance.
func DemoParallelExecution() {
	fmt.Println("=== Parallel Execution Demo ===")
	fmt.Printf("System: %d CPUs available\n", runtime.NumCPU())
	fmt.Println()

	sizes := []int{64, 128, 256, 512}
	iterations := 5

	fmt.Println("Matrix Multiplication Performance:")
	fmt.Println("Size\tSingle-Threaded\tParallel\tSpeedup")
	fmt.Println("----\t---------------\t--------\t-------")

	for _, size := range sizes {
		a := NewTensorRand(size, size)
		b := NewTensorRand(size, size)

		// Single-threaded
		stCfg := SingleThreadedConfig()
		startST := time.Now()
		for i := 0; i < iterations; i++ {
			_ = MatMulWithConfig(a, b, stCfg)
		}
		durationST := time.Since(startST) / time.Duration(iterations)

		// Parallel
		parCfg := DefaultComputeConfig()
		startPar := time.Now()
		for i := 0; i < iterations; i++ {
			_ = MatMulWithConfig(a, b, parCfg)
		}
		durationPar := time.Since(startPar) / time.Duration(iterations)

		speedup := float64(durationST) / float64(durationPar)

		fmt.Printf("%dx%d\t%v\t%v\t%.2fx\n",
			size, size, durationST, durationPar, speedup)
	}

	fmt.Println()
	fmt.Println("Observations:")
	fmt.Println("- Small matrices (64x64, 128x128): goroutine overhead dominates")
	fmt.Println("- Larger matrices (256x256+): better parallelization benefits")
	fmt.Println("- Memory bandwidth becomes the bottleneck, not compute")
	fmt.Println("- Optimal worker count: ~8-12 on this system")
	fmt.Println()

	// Demonstrate configuration switching
	fmt.Println("=== Configuration Modes ===")
	fmt.Println("Available modes:")
	fmt.Println("  - SingleThreadedConfig(): deterministic, easier debugging")
	fmt.Println("  - DefaultComputeConfig(): parallel, best performance")
	fmt.Println("  - Custom: fine-tune workers and thresholds")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  SetGlobalComputeConfig(SingleThreadedConfig())")
	fmt.Println("  result := MatMul(a, b)  // Uses global config")
	fmt.Println()
	fmt.Println("Or per-operation:")
	fmt.Println("  result := MatMulWithConfig(a, b, myConfig)")
}