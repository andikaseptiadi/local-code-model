//go:build linux && arm64

package main

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
)

// ===========================================================================
// SVE Vector Unit Tests for Graviton3/3E/4
// ===========================================================================
//
// Tests that exercise the different SVE configurations:
//   - Graviton3:  2× 256-bit SVE engines (4 float64 per vector)
//   - Graviton3E: 2× 256-bit SVE engines (35% higher throughput)
//   - Graviton4:  4× 128-bit SVE2 engines (2 float64 per vector)

// TestSVEVectorLength tests detection of SVE vector length
func TestSVEVectorLength(t *testing.T) {
	backend, err := NewSVEBackend()
	if err != nil {
		t.Skipf("SVE not available: %v", err)
	}

	generation := GetGravitonGeneration()
	vl := backend.VectorLength()
	hasSVE2 := backend.HasSVE2()
	features := DetectCPUFeatures()

	t.Logf("CPU: %s", GetCPUName())
	t.Logf("Graviton Generation: %d", generation)
	t.Logf("SVE Vector Length: %d float64 elements", vl)
	t.Logf("SVE Vector Width: %d bits", vl*64)
	t.Logf("SVE2 Support: %v", hasSVE2)
	t.Logf("Features: NEON=%v, SVE=%v, SVE2=%v", features.HasNEON, features.HasSVE, features.HasSVE2)

	// Verify expectations per generation
	switch generation {
	case 3:
		// Graviton3/3E: 256-bit SVE = 4 float64 elements
		if vl != 4 {
			t.Errorf("Graviton3 expected 4 elements (256-bit), got %d (%d-bit)", vl, vl*64)
		}
		if hasSVE2 {
			t.Errorf("Graviton3 should have SVE (not SVE2)")
		}
		t.Logf("✓ Graviton3: 256-bit SVE (4× float64), 2× vector engines")

	case 4:
		// Graviton4: Could be 128-bit (2 elements) or 512-bit (8 elements)
		// Most likely 128-bit per engine
		if vl < 2 || vl > 8 {
			t.Errorf("Graviton4 expected 2-8 elements, got %d", vl)
		}
		if !hasSVE2 {
			t.Errorf("Graviton4 should have SVE2")
		}
		t.Logf("✓ Graviton4: %d-bit SVE2 (%d× float64), 4× vector engines", vl*64, vl)
		if vl == 2 {
			t.Logf("  Note: 128-bit vectors, but 4× engines compensate")
		}

	default:
		t.Logf("Unknown Graviton generation or non-Graviton")
	}
}

// TestSVEVectorUnitCount attempts to detect number of SVE engines
func TestSVEVectorUnitCount(t *testing.T) {
	backend, err := NewSVEBackend()
	if err != nil {
		t.Skipf("SVE not available: %v", err)
	}

	generation := GetGravitonGeneration()
	hasSVE2 := backend.HasSVE2()

	var expectedEngines int
	var architecture string
	switch generation {
	case 3:
		expectedEngines = 2
		architecture = "Neoverse V1"
		if hasSVE2 {
			t.Errorf("Graviton3 should have SVE, not SVE2")
		}
		t.Logf("Graviton3/3E (%s): 2× 256-bit SVE engines", architecture)
	case 4:
		expectedEngines = 4
		architecture = "Neoverse V2"
		if !hasSVE2 {
			t.Errorf("Graviton4 should have SVE2")
		}
		t.Logf("Graviton4 (%s): 4× 128-bit SVE2 engines", architecture)
	default:
		t.Skipf("Not running on Graviton3/3E/4")
	}

	// We can't directly query engine count, but we can infer from parallelism
	// This is just for documentation
	t.Logf("Expected vector engines: %d", expectedEngines)
	t.Logf("CPU cores: %d", runtime.NumCPU())
	t.Logf("SVE version: %s", map[bool]string{true: "SVE2", false: "SVE"}[hasSVE2])
}

// TestSVEMultiThreadPerformance tests SVE performance across multiple threads
// This helps verify that multiple SVE engines are being utilized
func TestSVEMultiThreadPerformance(t *testing.T) {
	backend, err := NewSVEBackend()
	if err != nil {
		t.Skipf("SVE not available: %v", err)
	}

	generation := GetGravitonGeneration()
	if generation < 3 {
		t.Skipf("Need Graviton3+ for SVE")
	}

	size := 256
	a := NewTensorRand(size, size)
	b := NewTensorRand(size, size)

	// Single-threaded baseline
	result1, err := backend.MatMul(a, b)
	if err != nil {
		t.Fatalf("SVE MatMul failed: %v", err)
	}

	// Multi-threaded (simulate work across vector engines)
	// We'll split the work across goroutines
	numWorkers := 4
	results := make([]*Tensor, numWorkers)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			r, _ := backend.MatMul(a, b)
			results[idx] = r
		}(i)
	}
	wg.Wait()

	// Verify all results match
	for i, r := range results {
		maxDiff := 0.0
		for j := range r.data {
			diff := math.Abs(r.data[j] - result1.data[j])
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if maxDiff > 1e-10 {
			t.Errorf("Result %d differs: max diff %e", i, maxDiff)
		}
	}

	t.Logf("✓ All %d parallel SVE operations produced consistent results", numWorkers)
	t.Logf("  This exercises multiple SVE vector units on Graviton3/4")
}

// BenchmarkSVEVectorUtilization benchmarks SVE with different matrix sizes
// to understand how well the vector units are utilized
func BenchmarkSVEVectorUtilization(b *testing.B) {
	backend, err := NewSVEBackend()
	if err != nil {
		b.Skipf("SVE not available: %v", err)
	}

	generation := GetGravitonGeneration()
	vl := backend.VectorLength()

	b.Logf("Graviton%d: SVE with %d-element vectors (%d-bit)",
		generation, vl, vl*64)

	// Test sizes that are multiples of vector length
	sizes := []int{
		vl,      // Exact vector size
		vl * 2,  // 2 vectors
		vl * 4,  // 4 vectors
		vl * 8,  // 8 vectors
		64,      // Common size
		128,     // Moderate size
		256,     // Large size
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}
		})
	}
}

// BenchmarkSVEGravitonComparison compares SVE performance across Graviton generations
func BenchmarkSVEGravitonComparison(b *testing.B) {
	backend, err := NewSVEBackend()
	if err != nil {
		b.Skipf("SVE not available: %v", err)
	}

	generation := GetGravitonGeneration()
	if generation < 3 {
		b.Skipf("Need Graviton3+ for SVE")
	}

	vl := backend.VectorLength()
	hasSVE2 := backend.HasSVE2()
	sveVersion := "SVE"
	if hasSVE2 {
		sveVersion = "SVE2"
	}

	b.Logf("=== Graviton%d %s Benchmark ===", generation, sveVersion)
	b.Logf("Vector Length: %d elements (%d-bit)", vl, vl*64)

	switch generation {
	case 3:
		b.Logf("Architecture: 2× 256-bit SVE engines (Neoverse V1)")
		b.Logf("Expected: High performance with 4-element vectors")
		if hasSVE2 {
			b.Errorf("Graviton3 should have SVE, not SVE2")
		}
	case 4:
		b.Logf("Architecture: 4× 128-bit SVE2 engines (Neoverse V2)")
		b.Logf("Expected: More engines but narrower vectors")
		if !hasSVE2 {
			b.Errorf("Graviton4 should have SVE2")
		}
	}

	sizes := []int{128, 256, 512, 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%s_%d", sveVersion, size), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = backend.MatMul(a, mat)
			}

			// Calculate GFLOPS
			ops := 2.0 * float64(size) * float64(size) * float64(size) // 2 ops per multiply-add
			seconds := b.Elapsed().Seconds()
			gflops := (ops * float64(b.N)) / seconds / 1e9

			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkSVEEngineUtilization tests parallel utilization of SVE engines
func BenchmarkSVEEngineUtilization(b *testing.B) {
	backend, err := NewSVEBackend()
	if err != nil {
		b.Skipf("SVE not available: %v", err)
	}

	generation := GetGravitonGeneration()
	if generation < 3 {
		b.Skipf("Need Graviton3+ for SVE")
	}

	size := 256

	// Single-threaded baseline
	b.Run("Single", func(b *testing.B) {
		a := NewTensorRand(size, size)
		mat := NewTensorRand(size, size)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = backend.MatMul(a, mat)
		}
	})

	// Multi-threaded to utilize multiple vector engines
	parallelCounts := []int{2, 4, 8}
	for _, count := range parallelCounts {
		b.Run(fmt.Sprintf("Parallel_%d", count), func(b *testing.B) {
			a := NewTensorRand(size, size)
			mat := NewTensorRand(size, size)

			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					_, _ = backend.MatMul(a, mat)
				}
			})
		})
	}
}

// TestSVECorrectnessByGeneration tests correctness on each Graviton generation
func TestSVECorrectnessByGeneration(t *testing.T) {
	backend, err := NewSVEBackend()
	if err != nil {
		t.Skipf("SVE not available: %v", err)
	}

	generation := GetGravitonGeneration()
	vl := backend.VectorLength()

	t.Logf("Testing SVE on Graviton%d (VL=%d, %d-bit vectors)",
		generation, vl, vl*64)

	// Test various sizes
	sizes := []int{4, 8, 16, 32, 64, 128}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			a := NewTensorRand(size, size)
			b := NewTensorRand(size, size)

			// SVE result
			result, err := backend.MatMul(a, b)
			if err != nil {
				t.Fatalf("SVE MatMul failed: %v", err)
			}

			// Reference result
			expected := MatMul(a, b)

			// Compare
			maxDiff := 0.0
			for i := range result.data {
				diff := math.Abs(result.data[i] - expected.data[i])
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			tolerance := 1e-10
			if maxDiff > tolerance {
				t.Errorf("Size %d: result differs, max diff %e", size, maxDiff)
			}

			if size <= 32 {
				t.Logf("Size %d: max diff %e ✓", size, maxDiff)
			}
		})
	}
}
