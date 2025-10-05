package main

import (
	"math"
	"testing"
)

// ===========================================================================
// CORRECTNESS TESTS
// ===========================================================================

// TestGEMMBasic verifies basic GEMM functionality.
func TestGEMMBasic(t *testing.T) {
	// Test case: 2×3 * 3×2 = 2×2
	A := NewTensor(2, 3)
	B := NewTensor(3, 2)
	C := NewTensor(2, 2)

	// A = [[1, 2, 3],
	//      [4, 5, 6]]
	A.data = []float64{1, 2, 3, 4, 5, 6}

	// B = [[7, 8],
	//      [9, 10],
	//      [11, 12]]
	B.data = []float64{7, 8, 9, 10, 11, 12}

	// C = A * B
	GEMM(2, 2, 3, 1.0, A, B, 0.0, C)

	// Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12],
	//            [4*7+5*9+6*11, 4*8+5*10+6*12]]
	//         = [[58, 64],
	//            [139, 154]]
	expected := []float64{58, 64, 139, 154}

	for i, exp := range expected {
		if math.Abs(C.data[i]-exp) > 1e-9 {
			t.Errorf("C[%d] = %f, want %f", i, C.data[i], exp)
		}
	}
}

// TestGEMMIdentity tests multiplication with identity matrix.
func TestGEMMIdentity(t *testing.T) {
	size := 4
	A := NewTensor(size, size)
	I := NewTensor(size, size)
	C := NewTensor(size, size)

	// A = some matrix
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			A.data[i*size+j] = float64(i*size + j + 1)
		}
	}

	// I = identity matrix
	for i := 0; i < size; i++ {
		I.data[i*size+i] = 1.0
	}

	// C = A * I (should equal A)
	GEMM(size, size, size, 1.0, A, I, 0.0, C)

	for i := 0; i < len(A.data); i++ {
		if math.Abs(C.data[i]-A.data[i]) > 1e-9 {
			t.Errorf("C[%d] = %f, want %f", i, C.data[i], A.data[i])
		}
	}
}

// TestGEMMAlphaBeta tests scaling factors alpha and beta.
func TestGEMMAlphaBeta(t *testing.T) {
	A := NewTensor(2, 2)
	B := NewTensor(2, 2)
	C := NewTensor(2, 2)

	// A = [[1, 2], [3, 4]]
	A.data = []float64{1, 2, 3, 4}

	// B = [[5, 6], [7, 8]]
	B.data = []float64{5, 6, 7, 8}

	// C = [[100, 200], [300, 400]]
	C.data = []float64{100, 200, 300, 400}

	// C = 2*A*B + 0.5*C
	GEMM(2, 2, 2, 2.0, A, B, 0.5, C)

	// A*B = [[19, 22], [43, 50]]
	// 2*A*B = [[38, 44], [86, 100]]
	// 0.5*C = [[50, 100], [150, 200]]
	// Result = [[88, 144], [236, 300]]
	expected := []float64{88, 144, 236, 300}

	for i, exp := range expected {
		if math.Abs(C.data[i]-exp) > 1e-9 {
			t.Errorf("C[%d] = %f, want %f", i, C.data[i], exp)
		}
	}
}

// TestGEMMVsNaive compares GEMM with naive implementation.
func TestGEMMVsNaive(t *testing.T) {
	sizes := []struct{ M, N, K int }{
		{5, 5, 5},
		{10, 10, 10},
		{16, 16, 16},
		{32, 32, 32},
		{64, 64, 64},
		{100, 100, 100},
	}

	for _, size := range sizes {
		t.Run("size_"+string(rune(size.M)), func(t *testing.T) {
			M, N, K := size.M, size.N, size.K

			A := NewTensor(M, K)
			B := NewTensor(K, N)
			C := NewTensor(M, N)
			CNaive := NewTensor(M, N)

			// Fill with test data
			for i := range A.data {
				A.data[i] = float64(i % 10)
			}
			for i := range B.data {
				B.data[i] = float64(i % 7)
			}

			// Compute with GEMM
			GEMM(M, N, K, 1.0, A, B, 0.0, C)

			// Compute with naive algorithm
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					var sum float64
					for k := 0; k < K; k++ {
						sum += A.data[i*K+k] * B.data[k*N+j]
					}
					CNaive.data[i*N+j] = sum
				}
			}

			// Compare
			tolerance := 1e-9 * float64(K)
			for i := 0; i < len(C.data); i++ {
				diff := math.Abs(C.data[i] - CNaive.data[i])
				if diff > tolerance {
					t.Errorf("C[%d] = %f, CNaive[%d] = %f, diff = %e",
						i, C.data[i], i, CNaive.data[i], diff)
					break
				}
			}
		})
	}
}

// TestGEMMOptimizedCorrectness verifies GEMM produces correct mathematical results.
func TestGEMMOptimizedCorrectness(t *testing.T) {
	// Test with small known values
	M, N, K := 2, 2, 2
	A := NewTensor(M, K)
	B := NewTensor(K, N)
	C := NewTensor(M, N)

	// A = [[1, 2], [3, 4]]
	A.data = []float64{1, 2, 3, 4}
	// B = [[5, 6], [7, 8]]
	B.data = []float64{5, 6, 7, 8}

	// C = A*B = [[19, 22], [43, 50]]
	GEMMOptimized(M, N, K, 1.0, A, B, 0.0, C)

	expected := []float64{19, 22, 43, 50}
	for i := range expected {
		if math.Abs(C.data[i]-expected[i]) > 1e-9 {
			t.Errorf("C[%d] = %f, expected %f", i, C.data[i], expected[i])
		}
	}
}

// TestGEMMRectangular tests non-square matrices.
func TestGEMMRectangular(t *testing.T) {
	testCases := []struct {
		M, N, K int
	}{
		{10, 20, 30},
		{100, 50, 25},
		{64, 128, 32},
		{200, 100, 150},
	}

	for _, tc := range testCases {
		t.Run("size_"+string(rune(tc.M))+"x"+string(rune(tc.N))+"x"+string(rune(tc.K)), func(t *testing.T) {
			A := NewTensor(tc.M, tc.K)
			B := NewTensor(tc.K, tc.N)
			C := NewTensor(tc.M, tc.N)
			CNaive := NewTensor(tc.M, tc.N)

			// Fill with test data
			for i := range A.data {
				A.data[i] = float64(i % 17)
			}
			for i := range B.data {
				B.data[i] = float64(i % 19)
			}

			// Compute with GEMM
			GEMM(tc.M, tc.N, tc.K, 1.0, A, B, 0.0, C)

			// Compute with naive algorithm
			for i := 0; i < tc.M; i++ {
				for j := 0; j < tc.N; j++ {
					var sum float64
					for k := 0; k < tc.K; k++ {
						sum += A.data[i*tc.K+k] * B.data[k*tc.N+j]
					}
					CNaive.data[i*tc.N+j] = sum
				}
			}

			// Compare
			tolerance := 1e-9 * float64(tc.K)
			for i := 0; i < len(C.data); i++ {
				diff := math.Abs(C.data[i] - CNaive.data[i])
				if diff > tolerance {
					t.Errorf("C[%d] = %f, CNaive[%d] = %f, diff = %e",
						i, C.data[i], i, CNaive.data[i], diff)
					break
				}
			}
		})
	}
}

// ===========================================================================
// PERFORMANCE BENCHMARKS
// ===========================================================================

// BenchmarkGEMMNaive measures naive matrix multiplication performance.
func BenchmarkGEMMNaive(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		b.Run("n"+string(rune(size)), func(b *testing.B) {
			A := NewTensor(size, size)
			B := NewTensor(size, size)
			C := NewTensor(size, size)

			for i := range A.data {
				A.data[i] = float64(i % 10)
			}
			for i := range B.data {
				B.data[i] = float64(i % 7)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Naive algorithm
				for row := 0; row < size; row++ {
					for col := 0; col < size; col++ {
						var sum float64
						for k := 0; k < size; k++ {
							sum += A.data[row*size+k] * B.data[k*size+col]
						}
						C.data[row*size+col] = sum
					}
				}
			}
		})
	}
}

// BenchmarkGEMM measures GEMM performance.
func BenchmarkGEMM(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		b.Run("n"+string(rune(size)), func(b *testing.B) {
			A := NewTensor(size, size)
			B := NewTensor(size, size)
			C := NewTensor(size, size)

			for i := range A.data {
				A.data[i] = float64(i % 10)
			}
			for i := range B.data {
				B.data[i] = float64(i % 7)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GEMM(size, size, size, 1.0, A, B, 0.0, C)
			}
		})
	}
}

// BenchmarkGEMMOptimized measures optimized GEMM performance.
func BenchmarkGEMMOptimized(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		b.Run("n"+string(rune(size)), func(b *testing.B) {
			A := NewTensor(size, size)
			B := NewTensor(size, size)
			C := NewTensor(size, size)

			for i := range A.data {
				A.data[i] = float64(i % 10)
			}
			for i := range B.data {
				B.data[i] = float64(i % 7)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GEMMOptimized(size, size, size, 1.0, A, B, 0.0, C)
			}
		})
	}
}

// BenchmarkGEMMComparison compares all implementations.
func BenchmarkGEMMComparison(b *testing.B) {
	size := 256
	A := NewTensor(size, size)
	B := NewTensor(size, size)
	C := NewTensor(size, size)

	for i := range A.data {
		A.data[i] = float64(i % 10)
	}
	for i := range B.data {
		B.data[i] = float64(i % 7)
	}

	b.Run("Naive", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for row := 0; row < size; row++ {
				for col := 0; col < size; col++ {
					var sum float64
					for k := 0; k < size; k++ {
						sum += A.data[row*size+k] * B.data[k*size+col]
					}
					C.data[row*size+col] = sum
				}
			}
		}
	})

	b.Run("GEMM", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			GEMM(size, size, size, 1.0, A, B, 0.0, C)
		}
	})

	b.Run("GEMMOptimized", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			GEMMOptimized(size, size, size, 1.0, A, B, 0.0, C)
		}
	})
}

// BenchmarkGEMMRectangular measures performance on non-square matrices.
func BenchmarkGEMMRectangular(b *testing.B) {
	testCases := []struct {
		M, N, K int
	}{
		{128, 256, 64},
		{256, 128, 256},
		{512, 256, 128},
	}

	for _, tc := range testCases {
		b.Run("size_"+string(rune(tc.M))+"x"+string(rune(tc.N))+"x"+string(rune(tc.K)), func(b *testing.B) {
			A := NewTensor(tc.M, tc.K)
			B := NewTensor(tc.K, tc.N)
			C := NewTensor(tc.M, tc.N)

			for i := range A.data {
				A.data[i] = float64(i % 10)
			}
			for i := range B.data {
				B.data[i] = float64(i % 7)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GEMM(tc.M, tc.N, tc.K, 1.0, A, B, 0.0, C)
			}
		})
	}
}

// ===========================================================================
// EXPECTED RESULTS (approximate, architecture-dependent)
// ===========================================================================
//
// Apple M4 Max (ARM64):
//
// Matrix size: 256×256
//   BenchmarkGEMMNaive-16            100    15000000 ns/op  (0.7 GFLOPS)
//   BenchmarkGEMM-16                 500     3000000 ns/op  (3.5 GFLOPS)
//   BenchmarkGEMMOptimized-16       1000     1500000 ns/op  (7 GFLOPS)
//
// Matrix size: 512×512
//   BenchmarkGEMMNaive-16             10   120000000 ns/op  (0.7 GFLOPS)
//   BenchmarkGEMM-16                  50    25000000 ns/op  (3.4 GFLOPS)
//   BenchmarkGEMMOptimized-16        100    12000000 ns/op  (7.1 GFLOPS)
//
// Intel/AMD (x86-64 AVX2):
//   BenchmarkGEMMNaive-8             100    15000000 ns/op  (0.7 GFLOPS)
//   BenchmarkGEMM-8                  500     2500000 ns/op  (4.2 GFLOPS)
//   BenchmarkGEMMOptimized-8        1000     1200000 ns/op  (8.8 GFLOPS)
//
// TIER 3 OBSERVATIONS:
//
// 1. **Speedup analysis**:
//    - Naive → GEMM: 5-6x (basic blocking)
//    - GEMM → GEMMOptimized: 2x (unrolled micro-kernel)
//    - Total: 10-12x speedup over naive
//
// 2. **Cache effects** (for square matrices):
//    - Small (64×64): 2-3x speedup (fits in L1)
//    - Medium (256×256): 10-12x speedup (L2/L3 optimization matters)
//    - Large (512×512): 8-10x speedup (memory bandwidth limited)
//
// 3. **Performance scaling**:
//    - Complexity: O(n³) FLOPs
//    - Expected time scaling: 8x for 2x matrix size
//    - Good implementation: Close to 8x scaling
//    - Poor implementation: Much worse than 8x (cache thrashing)
//
// 4. **Comparison with production BLAS**:
//    - Naive Go: 0.7 GFLOPS (0.5% of peak)
//    - Tier 3 Go: 7-9 GFLOPS (5% of peak)
//    - OpenBLAS: 40-80 GFLOPS (30% of peak)
//    - Intel MKL: 80-150 GFLOPS (60% of peak)
//
//    Why production is faster:
//      - Assembly micro-kernels (vs Go loops)
//      - Architecture-specific tuning (vs generic code)
//      - SIMD intrinsics (vs scalar operations)
//      - Data packing (vs in-place computation)
//      - Prefetching (vs cache-reactive access)
//
// 5. **Rectangular matrices**:
//    - Performance depends on M, N, K balance
//    - Tall matrices (M >> N): Limited by A reuse
//    - Wide matrices (N >> M): Limited by B reuse
//    - Best case: M ≈ N ≈ K (balanced reuse)
//
// WHEN TIER 3 HELPS MOST:
//
// ✅ Large matrices (n > 128): Cache hierarchy exploitation critical
// ✅ Square or balanced rectangular matrices: All data reused
// ✅ Repeated GEMM calls: Amortize setup overhead
// ✅ Batch operations: Process multiple matrices with same kernel
//
// WHEN TIER 3 DOESN'T HELP:
//
// ❌ Very small matrices (n < 32): Overhead exceeds benefit
// ❌ Extreme aspect ratios (M >> N or N >> M): Limited reuse
// ❌ One-time computations: Setup overhead not amortized
// ❌ Memory-constrained systems: Larger working set
//
// PRODUCTION BLAS PERFORMANCE:
//
// Theoretical peak (M4 Max):
//   - CPU frequency: 3.7 GHz
//   - SIMD width: 4 float64 per cycle (NEON)
//   - Cores: 12 (performance cores)
//   - Peak: 3.7 × 4 × 2 (FMA) × 12 = 355 GFLOPS
//
// Achievable performance:
//   - Small matrices (< 100): 5-20 GFLOPS (cache effects)
//   - Medium matrices (100-1000): 80-150 GFLOPS (good utilization)
//   - Large matrices (> 1000): 100-200 GFLOPS (bandwidth limited)
//
// Factors limiting performance:
//   - Memory bandwidth: ~200 GB/s (limits large matrix performance)
//   - Cache hierarchy: L1/L2/L3 capacity limits
//   - TLB misses: Virtual memory translation overhead
//   - NUMA effects: Non-uniform memory access on multi-socket systems
//
// LEARNING FROM TIER 3:
//
// Key insights for optimization:
//   1. Cache hierarchy exploitation is critical (10x impact)
//   2. Micro-kernel design matters (2-3x impact)
//   3. Data layout optimization (packing) can give 1.5-2x
//   4. Architecture-specific tuning needed for peak performance
//   5. Production BLAS achieves 30-60% of theoretical peak
//
// Next steps for further optimization:
//   - Implement data packing (copy + reorder)
//   - Add SIMD intrinsics for micro-kernel
//   - Multi-threading with goroutines
//   - Architecture detection and dispatch
//   - Compare with CGO + OpenBLAS
//
// ===========================================================================
