// ===========================================================================
// SVE Matrix Multiplication using C Intrinsics
// ===========================================================================
//
// ARM SVE (Scalable Vector Extension) implementation using official intrinsics.
// This provides true vector-length agnostic code that works on:
//   - Graviton3/3E: 256-bit SVE
//   - Graviton4: 512-bit SVE2
//
// Compile with: gcc -march=armv8.2-a+sve -O3
//
// ===========================================================================

#include <stdint.h>  // For int64_t (needed in both SVE and fallback)

#if defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

// matmul_sve_c performs matrix multiplication using SVE intrinsics
// C = A * B where A is m×k, B is k×n, C is m×n
void matmul_sve_c(double* restrict c, const double* restrict a,
                  const double* restrict b, int64_t m, int64_t n, int64_t k) {

    // Get SVE vector length in elements (double precision)
    const int64_t vl = svcntd();  // Vector length in float64 elements

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            svfloat64_t sum_vec = svdup_n_f64(0.0);  // Zero vector

            // Process k elements in chunks of vector length
            int64_t l = 0;
            for (; l + vl <= k; l += vl) {
                // Create predicate for all active lanes
                svbool_t pg = svptrue_b64();

                // Load vector from A[i][l:l+vl]
                svfloat64_t a_vec = svld1_f64(pg, &a[i * k + l]);

                // Load vector from B[l:l+vl][j]
                // This requires strided load (stride = n)
                svfloat64_t b_vec = svdup_n_f64(0.0);
                for (int64_t vl_idx = 0; vl_idx < vl; vl_idx++) {
                    b_vec = svdup_n_f64_lane(b_vec, vl_idx, b[(l + vl_idx) * n + j]);
                }

                // Fused multiply-add: sum_vec += a_vec * b_vec
                sum_vec = svmla_f64_m(pg, sum_vec, a_vec, b_vec);
            }

            // Horizontal reduction: sum all elements in sum_vec
            double sum = svaddv_f64(svptrue_b64(), sum_vec);

            // Handle remaining elements (l to k)
            for (; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }

            c[i * n + j] = sum;
        }
    }
}

// matmul_sve_optimized: Better version with improved memory access
// Uses block tiling for better cache locality
void matmul_sve_optimized(double* restrict c, const double* restrict a,
                          const double* restrict b, int64_t m, int64_t n, int64_t k) {

    const int64_t vl = svcntd();
    const int64_t BLOCK_SIZE = 64;  // Cache-friendly block size

    // Initialize C to zero
    for (int64_t i = 0; i < m * n; i++) {
        c[i] = 0.0;
    }

    // Blocked matrix multiplication
    for (int64_t ii = 0; ii < m; ii += BLOCK_SIZE) {
        for (int64_t jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int64_t kk = 0; kk < k; kk += BLOCK_SIZE) {

                // Process block
                int64_t i_max = (ii + BLOCK_SIZE < m) ? ii + BLOCK_SIZE : m;
                int64_t j_max = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;
                int64_t k_max = (kk + BLOCK_SIZE < k) ? kk + BLOCK_SIZE : k;

                for (int64_t i = ii; i < i_max; i++) {
                    for (int64_t j = jj; j < j_max; j++) {
                        svfloat64_t sum_vec = svdup_n_f64(0.0);

                        int64_t l = kk;
                        svbool_t pg = svptrue_b64();

                        // Vectorized inner loop
                        for (; l + vl <= k_max; l += vl) {
                            svfloat64_t a_vec = svld1_f64(pg, &a[i * k + l]);

                            // Gather B values (stride = n)
                            int64_t indices[8];  // Max VL for Graviton4
                            for (int64_t idx = 0; idx < vl && l + idx < k_max; idx++) {
                                indices[idx] = (l + idx) * n + j;
                            }
                            svfloat64_t b_vec = svld1_gather_u64index_f64(pg, b, svld1_u64(pg, (const uint64_t*)indices));

                            sum_vec = svmla_f64_m(pg, sum_vec, a_vec, b_vec);
                        }

                        c[i * n + j] += svaddv_f64(pg, sum_vec);

                        // Scalar remainder
                        for (; l < k_max; l++) {
                            c[i * n + j] += a[i * k + l] * b[l * n + j];
                        }
                    }
                }
            }
        }
    }
}

#else  // No SVE support

// Fallback scalar implementation
void matmul_sve_c(double* restrict c, const double* restrict a,
                  const double* restrict b, int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (int64_t l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void matmul_sve_optimized(double* restrict c, const double* restrict a,
                          const double* restrict b, int64_t m, int64_t n, int64_t k) {
    matmul_sve_c(c, a, b, m, n, k);  // Same as basic version
}

#endif

// Query SVE vector length
int64_t sve_vector_length(void) {
#if defined(__ARM_FEATURE_SVE)
    return (int64_t)svcntd();  // Elements per vector (float64)
#else
    return 0;  // SVE not available
#endif
}
