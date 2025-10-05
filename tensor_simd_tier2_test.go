package main

import (
	"math"
	"testing"
)

// ===========================================================================
// CORRECTNESS TESTS
// ===========================================================================

// TestDotProductUnrolled2Basic verifies 2x unrolled dot product correctness.
func TestDotProductUnrolled2Basic(t *testing.T) {
	testCases := []struct {
		name     string
		a, b     []float64
		expected float64
	}{
		{
			name:     "simple",
			a:        []float64{1, 2, 3, 4},
			b:        []float64{5, 6, 7, 8},
			expected: 1*5 + 2*6 + 3*7 + 4*8, // 70
		},
		{
			name:     "odd_length",
			a:        []float64{1, 2, 3, 4, 5},
			b:        []float64{5, 6, 7, 8, 9},
			expected: 1*5 + 2*6 + 3*7 + 4*8 + 5*9, // 115
		},
		{
			name:     "zeros",
			a:        []float64{0, 0, 0, 0},
			b:        []float64{1, 2, 3, 4},
			expected: 0,
		},
		{
			name:     "negative",
			a:        []float64{1, -2, 3, -4},
			b:        []float64{5, 6, 7, 8},
			expected: 1*5 + (-2)*6 + 3*7 + (-4)*8, // -20
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := DotProductUnrolled2(tc.a, tc.b)
			if math.Abs(got-tc.expected) > 1e-9 {
				t.Errorf("got %f, want %f", got, tc.expected)
			}
		})
	}
}

// TestDotProductUnrolled4Basic verifies 4x unrolled dot product correctness.
func TestDotProductUnrolled4Basic(t *testing.T) {
	testCases := []struct {
		name     string
		a, b     []float64
		expected float64
	}{
		{
			name:     "simple",
			a:        []float64{1, 2, 3, 4, 5, 6, 7, 8},
			b:        []float64{1, 1, 1, 1, 1, 1, 1, 1},
			expected: 36.0,
		},
		{
			name:     "remainder_1",
			a:        []float64{1, 2, 3, 4, 5},
			b:        []float64{1, 1, 1, 1, 1},
			expected: 15.0,
		},
		{
			name:     "remainder_2",
			a:        []float64{1, 2, 3, 4, 5, 6},
			b:        []float64{1, 1, 1, 1, 1, 1},
			expected: 21.0,
		},
		{
			name:     "remainder_3",
			a:        []float64{1, 2, 3, 4, 5, 6, 7},
			b:        []float64{1, 1, 1, 1, 1, 1, 1},
			expected: 28.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := DotProductUnrolled4(tc.a, tc.b)
			if math.Abs(got-tc.expected) > 1e-9 {
				t.Errorf("got %f, want %f", got, tc.expected)
			}
		})
	}
}

// TestDotProductUnrolled8Basic verifies 8x unrolled dot product correctness.
func TestDotProductUnrolled8Basic(t *testing.T) {
	testCases := []struct {
		name     string
		size     int
		expected float64
	}{
		{"size_8", 8, 36.0},
		{"size_9", 9, 45.0},
		{"size_15", 15, 120.0},
		{"size_16", 16, 136.0},
		{"size_17", 17, 153.0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := make([]float64, tc.size)
			b := make([]float64, tc.size)
			for i := range a {
				a[i] = float64(i + 1)
				b[i] = 1.0
			}

			got := DotProductUnrolled8(a, b)
			if math.Abs(got-tc.expected) > 1e-9 {
				t.Errorf("got %f, want %f", got, tc.expected)
			}
		})
	}
}

// TestDotProductUnrolledConsistency verifies all unrolled versions match.
func TestDotProductUnrolledConsistency(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1000}

	for _, size := range sizes {
		t.Run("size_"+string(rune(size)), func(t *testing.T) {
			a := make([]float64, size)
			b := make([]float64, size)

			for i := range a {
				a[i] = float64(i + 1)
				b[i] = float64(i*2 + 1)
			}

			// Compute reference with pure Go
			reference := dotProductGo(a, b)

			// Compare all unrolled versions
			r2 := DotProductUnrolled2(a, b)
			r4 := DotProductUnrolled4(a, b)
			r8 := DotProductUnrolled8(a, b)

			tolerance := 1e-9 * float64(size)

			if math.Abs(r2-reference) > tolerance {
				t.Errorf("Unrolled2 mismatch: got %f, want %f", r2, reference)
			}
			if math.Abs(r4-reference) > tolerance {
				t.Errorf("Unrolled4 mismatch: got %f, want %f", r4, reference)
			}
			if math.Abs(r8-reference) > tolerance {
				t.Errorf("Unrolled8 mismatch: got %f, want %f", r8, reference)
			}
		})
	}
}

// TestMatVecTiledBasic verifies register-tiled matrix-vector multiply.
func TestMatVecTiledBasic(t *testing.T) {
	// Create a simple 4×4 identity matrix
	A := NewTensor(4, 4)
	for i := 0; i < 4; i++ {
		A.data[i*4+i] = 1.0
	}

	x := []float64{1, 2, 3, 4}

	y := MatVecTiled(A, x)

	// Identity matrix: y should equal x
	for i := range x {
		if math.Abs(y[i]-x[i]) > 1e-9 {
			t.Errorf("y[%d] = %f, want %f", i, y[i], x[i])
		}
	}
}

// TestMatVecTiledCorrectness verifies correctness for various matrix sizes.
func TestMatVecTiledCorrectness(t *testing.T) {
	sizes := []struct{ m, n int }{
		{4, 4},
		{5, 5},
		{8, 8},
		{10, 10},
		{16, 16},
		{20, 20},
		{100, 100},
	}

	for _, size := range sizes {
		t.Run("size_"+string(rune(size.m))+"x"+string(rune(size.n)), func(t *testing.T) {
			m, n := size.m, size.n

			// Create test matrix and vector
			A := NewTensor(m, n)
			x := make([]float64, n)

			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					A.data[i*n+j] = float64(i*n + j + 1)
				}
			}
			for j := 0; j < n; j++ {
				x[j] = float64(j + 1)
			}

			// Compute reference (naive implementation)
			yRef := make([]float64, m)
			for i := 0; i < m; i++ {
				var sum float64
				for j := 0; j < n; j++ {
					sum += A.data[i*n+j] * x[j]
				}
				yRef[i] = sum
			}

			// Compute with tiled version
			y := MatVecTiled(A, x)

			// Compare
			tolerance := 1e-9 * float64(n)
			for i := 0; i < m; i++ {
				if math.Abs(y[i]-yRef[i]) > tolerance {
					t.Errorf("y[%d] = %f, want %f (diff = %e)", i, y[i], yRef[i], math.Abs(y[i]-yRef[i]))
				}
			}
		})
	}
}

// TestMatVecTiledUnrolledCorrectness verifies tiled+unrolled version.
func TestMatVecTiledUnrolledCorrectness(t *testing.T) {
	sizes := []struct{ m, n int }{
		{4, 4},
		{5, 5},
		{8, 8},
		{10, 10},
		{16, 16},
		{20, 20},
		{100, 100},
	}

	for _, size := range sizes {
		t.Run("size_"+string(rune(size.m))+"x"+string(rune(size.n)), func(t *testing.T) {
			m, n := size.m, size.n

			// Create test matrix and vector
			A := NewTensor(m, n)
			x := make([]float64, n)

			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					A.data[i*n+j] = float64(i*n + j + 1)
				}
			}
			for j := 0; j < n; j++ {
				x[j] = float64(j + 1)
			}

			// Compute reference
			yRef := make([]float64, m)
			for i := 0; i < m; i++ {
				var sum float64
				for j := 0; j < n; j++ {
					sum += A.data[i*n+j] * x[j]
				}
				yRef[i] = sum
			}

			// Compute with tiled+unrolled version
			y := MatVecTiledUnrolled(A, x)

			// Compare
			tolerance := 1e-9 * float64(n)
			for i := 0; i < m; i++ {
				if math.Abs(y[i]-yRef[i]) > tolerance {
					t.Errorf("y[%d] = %f, want %f (diff = %e)", i, y[i], yRef[i], math.Abs(y[i]-yRef[i]))
				}
			}
		})
	}
}

// ===========================================================================
// PERFORMANCE BENCHMARKS
// ===========================================================================

// BenchmarkDotProductUnrolled2 measures 2x unrolled performance.
func BenchmarkDotProductUnrolled2(b *testing.B) {
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		b.Run("n"+string(rune(size)), func(b *testing.B) {
			a := make([]float64, size)
			vec := make([]float64, size)
			for i := range a {
				a[i] = float64(i + 1)
				vec[i] = float64(i*2 + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = DotProductUnrolled2(a, vec)
			}
		})
	}
}

// BenchmarkDotProductUnrolled4 measures 4x unrolled performance.
func BenchmarkDotProductUnrolled4(b *testing.B) {
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		b.Run("n"+string(rune(size)), func(b *testing.B) {
			a := make([]float64, size)
			vec := make([]float64, size)
			for i := range a {
				a[i] = float64(i + 1)
				vec[i] = float64(i*2 + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = DotProductUnrolled4(a, vec)
			}
		})
	}
}

// BenchmarkDotProductUnrolled8 measures 8x unrolled performance.
func BenchmarkDotProductUnrolled8(b *testing.B) {
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		b.Run("n"+string(rune(size)), func(b *testing.B) {
			a := make([]float64, size)
			vec := make([]float64, size)
			for i := range a {
				a[i] = float64(i + 1)
				vec[i] = float64(i*2 + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = DotProductUnrolled8(a, vec)
			}
		})
	}
}

// BenchmarkDotProductUnrollComparison compares different unroll factors.
func BenchmarkDotProductUnrollComparison(b *testing.B) {
	size := 1000
	a := make([]float64, size)
	vec := make([]float64, size)
	for i := range a {
		a[i] = float64(i + 1)
		vec[i] = float64(i*2 + 1)
	}

	b.Run("Baseline", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = dotProductGo(a, vec)
		}
	})

	b.Run("Unroll2", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = DotProductUnrolled2(a, vec)
		}
	})

	b.Run("Unroll4", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = DotProductUnrolled4(a, vec)
		}
	})

	b.Run("Unroll8", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = DotProductUnrolled8(a, vec)
		}
	})
}

// BenchmarkMatVecTiled measures register-tiled matrix-vector multiply.
func BenchmarkMatVecTiled(b *testing.B) {
	sizes := []struct{ m, n int }{
		{100, 100},
		{500, 500},
		{1000, 1000},
	}

	for _, size := range sizes {
		b.Run("size_"+string(rune(size.m))+"x"+string(rune(size.n)), func(b *testing.B) {
			m, n := size.m, size.n
			A := NewTensor(m, n)
			x := make([]float64, n)

			for i := range A.data {
				A.data[i] = float64(i + 1)
			}
			for i := range x {
				x[i] = float64(i + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatVecTiled(A, x)
			}
		})
	}
}

// BenchmarkMatVecTiledUnrolled measures tiled+unrolled matrix-vector multiply.
func BenchmarkMatVecTiledUnrolled(b *testing.B) {
	sizes := []struct{ m, n int }{
		{100, 100},
		{500, 500},
		{1000, 1000},
	}

	for _, size := range sizes {
		b.Run("size_"+string(rune(size.m))+"x"+string(rune(size.n)), func(b *testing.B) {
			m, n := size.m, size.n
			A := NewTensor(m, n)
			x := make([]float64, n)

			for i := range A.data {
				A.data[i] = float64(i + 1)
			}
			for i := range x {
				x[i] = float64(i + 1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MatVecTiledUnrolled(A, x)
			}
		})
	}
}

// BenchmarkMatVecComparison compares naive vs tiled vs tiled+unrolled.
func BenchmarkMatVecComparison(b *testing.B) {
	m, n := 1000, 1000
	A := NewTensor(m, n)
	x := make([]float64, n)

	for i := range A.data {
		A.data[i] = float64(i + 1)
	}
	for i := range x {
		x[i] = float64(i + 1)
	}

	b.Run("Naive", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			y := make([]float64, m)
			for row := 0; row < m; row++ {
				var sum float64
				for col := 0; col < n; col++ {
					sum += A.data[row*n+col] * x[col]
				}
				y[row] = sum
			}
			_ = y
		}
	})

	b.Run("Tiled", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = MatVecTiled(A, x)
		}
	})

	b.Run("TiledUnrolled", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = MatVecTiledUnrolled(A, x)
		}
	})
}

// ===========================================================================
// EXPECTED RESULTS (approximate, architecture-dependent)
// ===========================================================================
//
// Apple M4 Max (ARM64):
//
// Dot Product Unrolling (n=1000):
//   BenchmarkDotProductGo-8              100000    10000 ns/op  (baseline)
//   BenchmarkDotProductUnrolled2-8       120000     8500 ns/op  (1.2x speedup)
//   BenchmarkDotProductUnrolled4-8       150000     7000 ns/op  (1.4x speedup)
//   BenchmarkDotProductUnrolled8-8       160000     6500 ns/op  (1.5x speedup)
//
// Intel/AMD (x86-64):
//   BenchmarkDotProductGo-8              100000    10000 ns/op  (baseline)
//   BenchmarkDotProductUnrolled2-8       130000     7500 ns/op  (1.3x speedup)
//   BenchmarkDotProductUnrolled4-8       170000     6000 ns/op  (1.7x speedup)
//   BenchmarkDotProductUnrolled8-8       180000     5500 ns/op  (1.8x speedup)
//
// Matrix-Vector (1000×1000):
//   BenchmarkMatVecNaive-8                  500  3000000 ns/op  (baseline)
//   BenchmarkMatVecTiled-8                 1500  1000000 ns/op  (3x speedup)
//   BenchmarkMatVecTiledUnrolled-8         2000   700000 ns/op  (4.3x speedup)
//
// TIER 2 OBSERVATIONS:
//
// 1. **Unrolling effectiveness**: Varies by architecture
//    - x86-64: Better ILP, more benefit from unrolling
//    - ARM64: Lower benefit due to in-order execution (some ARM cores)
//    - Diminishing returns beyond 4-8x unrolling
//
// 2. **Register tiling**: Highly effective for matrix operations
//    - 3-4x speedup from cache locality and register reuse
//    - Particularly good for tall matrices (many rows, few columns)
//    - Combining with unrolling gives 4-5x total speedup
//
// 3. **Memory bandwidth**: Limiting factor for large vectors
//    - Dot product: 1.5-2x speedup (bandwidth-bound)
//    - Matrix-vector: 3-4x speedup (compute-bound with good locality)
//
// 4. **Compiler optimization**: Go compiler may help or hurt
//    - Sometimes auto-vectorizes simple loops
//    - Bounds checking can limit effectiveness
//    - Use -gcflags="-B" to disable bounds checks for max performance
//
// WHEN TIER 2 HELPS MOST:
//
// ✅ Matrix operations (high arithmetic intensity)
// ✅ Medium to large sizes (100-10000 elements)
// ✅ Regular access patterns (sequential, strided)
// ✅ Multiple independent operations (ILP opportunities)
//
// WHEN TIER 2 DOESN'T HELP:
//
// ❌ Very small operations (overhead dominates)
// ❌ Memory-bound operations (bandwidth saturated)
// ❌ Irregular access patterns (scatter/gather)
// ❌ Code with branches (limits unrolling effectiveness)
//
// NEXT STEPS (Tier 3):
//
// - BLAS-style micro-kernels (cache-aware blocking)
// - Panel-panel matrix multiplication (GEMM)
// - Recursive decomposition (cache-oblivious algorithms)
// - Architecture-specific tuning (AVX-512, SVE)
//
// ===========================================================================
