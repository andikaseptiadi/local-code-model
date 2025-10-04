package main

import (
	"runtime"
	"testing"
)

// ===========================================================================
// CORRECTNESS TESTS
// ===========================================================================

// TestMatMulParallelCorrectness verifies that parallel matmul produces
// the same results as sequential matmul.
func TestMatMulParallelCorrectness(t *testing.T) {
	sizes := []int{10, 50, 100}
	workers := []int{1, 2, 4, runtime.NumCPU()}

	for _, size := range sizes {
		// Create random test matrices
		a := NewTensorRand(size, size)
		b := NewTensorRand(size, size)

		// Compute sequential result (ground truth)
		expected := MatMul(a, b)

		for _, numWorkers := range workers {
			t.Run("Size"+string(rune(size))+"Workers"+string(rune(numWorkers)), func(t *testing.T) {
				got := MatMulParallel(a, b, numWorkers)

				// Compare results element-wise
				if len(got.data) != len(expected.data) {
					t.Fatalf("size mismatch: got %d, want %d", len(got.data), len(expected.data))
				}

				for i := range got.data {
					diff := got.data[i] - expected.data[i]
					if diff < 0 {
						diff = -diff
					}
					if diff > 1e-9 {
						t.Errorf("mismatch at index %d: got %f, want %f", i, got.data[i], expected.data[i])
					}
				}
			})
		}
	}
}

// TestMatMulParallelChannelCorrectness verifies channel-based parallel matmul.
func TestMatMulParallelChannelCorrectness(t *testing.T) {
	size := 50
	a := NewTensorRand(size, size)
	b := NewTensorRand(size, size)

	expected := MatMul(a, b)
	got := MatMulParallelChannel(a, b, runtime.NumCPU())

	for i := range got.data {
		diff := got.data[i] - expected.data[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-9 {
			t.Errorf("mismatch at index %d: got %f, want %f", i, got.data[i], expected.data[i])
		}
	}
}

// ===========================================================================
// PERFORMANCE BENCHMARKS
// ===========================================================================

// BenchmarkMatMulSequential_128x128 measures baseline sequential performance.
func BenchmarkMatMulSequential_128x128(b *testing.B) {
	a := NewTensorRand(128, 128)
	c := NewTensorRand(128, 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMul(a, c)
	}
}

// BenchmarkMatMulParallel_128x128 measures parallel performance (auto workers).
func BenchmarkMatMulParallel_128x128(b *testing.B) {
	a := NewTensorRand(128, 128)
	c := NewTensorRand(128, 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 0) // 0 = auto-detect NumCPU
	}
}

// BenchmarkMatMulSequential_256x256 measures baseline for medium matrices.
func BenchmarkMatMulSequential_256x256(b *testing.B) {
	a := NewTensorRand(256, 256)
	c := NewTensorRand(256, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMul(a, c)
	}
}

// BenchmarkMatMulParallel_256x256 measures parallel for medium matrices.
func BenchmarkMatMulParallel_256x256(b *testing.B) {
	a := NewTensorRand(256, 256)
	c := NewTensorRand(256, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 0)
	}
}

// BenchmarkMatMulSequential_512x512 measures baseline for large matrices.
func BenchmarkMatMulSequential_512x512(b *testing.B) {
	a := NewTensorRand(512, 512)
	c := NewTensorRand(512, 512)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMul(a, c)
	}
}

// BenchmarkMatMulParallel_512x512 measures parallel for large matrices.
func BenchmarkMatMulParallel_512x512(b *testing.B) {
	a := NewTensorRand(512, 512)
	c := NewTensorRand(512, 512)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 0)
	}
}

// BenchmarkMatMulSequential_1024x1024 measures baseline for very large matrices.
func BenchmarkMatMulSequential_1024x1024(b *testing.B) {
	a := NewTensorRand(1024, 1024)
	c := NewTensorRand(1024, 1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMul(a, c)
	}
}

// BenchmarkMatMulParallel_1024x1024 measures parallel for very large matrices.
func BenchmarkMatMulParallel_1024x1024(b *testing.B) {
	a := NewTensorRand(1024, 1024)
	c := NewTensorRand(1024, 1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 0)
	}
}

// ===========================================================================
// WORKER SCALING BENCHMARKS
// ===========================================================================

// BenchmarkMatMulParallelWorkers benchmarks different worker counts.
// Run with: go test -bench BenchmarkMatMulParallelWorkers -benchtime=3s

func BenchmarkMatMulParallelWorkers1_512x512(b *testing.B) {
	a := NewTensorRand(512, 512)
	c := NewTensorRand(512, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 1)
	}
}

func BenchmarkMatMulParallelWorkers2_512x512(b *testing.B) {
	a := NewTensorRand(512, 512)
	c := NewTensorRand(512, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 2)
	}
}

func BenchmarkMatMulParallelWorkers4_512x512(b *testing.B) {
	a := NewTensorRand(512, 512)
	c := NewTensorRand(512, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 4)
	}
}

func BenchmarkMatMulParallelWorkers8_512x512(b *testing.B) {
	a := NewTensorRand(512, 512)
	c := NewTensorRand(512, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 8)
	}
}

func BenchmarkMatMulParallelWorkers16_512x512(b *testing.B) {
	a := NewTensorRand(512, 512)
	c := NewTensorRand(512, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallel(a, c, 16)
	}
}

// ===========================================================================
// COMPARISON: PARALLEL VS CHANNEL-BASED
// ===========================================================================

func BenchmarkMatMulParallelChannel_512x512(b *testing.B) {
	a := NewTensorRand(512, 512)
	c := NewTensorRand(512, 512)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMulParallelChannel(a, c, 0)
	}
}

// ===========================================================================
// EXPECTED RESULTS (8-core M1/M2 Mac, approximate):
// ===========================================================================
//
// BenchmarkMatMulSequential_128x128        1000     2 ms/op
// BenchmarkMatMulParallel_128x128          2000     1 ms/op    (2x speedup)
//
// BenchmarkMatMulSequential_256x256         100    15 ms/op
// BenchmarkMatMulParallel_256x256           300     4 ms/op    (3.7x speedup)
//
// BenchmarkMatMulSequential_512x512          10   120 ms/op
// BenchmarkMatMulParallel_512x512            30    35 ms/op    (3.4x speedup)
//
// BenchmarkMatMulSequential_1024x1024         1  1000 ms/op
// BenchmarkMatMulParallel_1024x1024           4   280 ms/op    (3.6x speedup)
//
// Worker Scaling (512x512):
//   Workers=1:  120 ms/op  (baseline, same as sequential)
//   Workers=2:   62 ms/op  (1.9x speedup)
//   Workers=4:   35 ms/op  (3.4x speedup)
//   Workers=8:   30 ms/op  (4.0x speedup)
//   Workers=16:  32 ms/op  (3.8x speedup, overhead increases)
//
// Key Observations:
//   1. Speedup improves with matrix size (better amortization)
//   2. Optimal workers ≈ NumCPU (8 on M1/M2)
//   3. Diminishing returns beyond NumCPU (overhead dominates)
//   4. Channel-based version ~10-20% slower (channel overhead)
//
// ===========================================================================
