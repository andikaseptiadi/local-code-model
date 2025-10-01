// ===========================================================================
// NEON Matrix Multiplication using C Intrinsics
// ===========================================================================
//
// ARM NEON implementation for matrix multiplication on Linux ARM64.
// NEON is mandatory on ARM64, so this works on all Graviton generations.
//
// Compile with: gcc -march=armv8-a+simd -O3
//
// ===========================================================================

#include <arm_neon.h>
#include <stdint.h>

// matmul_neon_c performs matrix multiplication using NEON intrinsics
// C = A * B where A is m×k, B is k×n, C is m×n
void matmul_neon_c(double* restrict c, const double* restrict a,
                   const double* restrict b, int64_t m, int64_t n, int64_t k) {

    // NEON processes 2 float64 elements at a time (128-bit vectors)
    const int64_t NEON_WIDTH = 2;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float64x2_t sum_vec = vdupq_n_f64(0.0);  // Zero vector

            // Process k elements in chunks of 2
            int64_t l = 0;
            for (; l + NEON_WIDTH <= k; l += NEON_WIDTH) {
                // Load 2 elements from A[i][l:l+2]
                float64x2_t a_vec = vld1q_f64(&a[i * k + l]);

                // Load 2 elements from B[l:l+2][j]
                // This requires two separate loads with stride n
                double b_vals[2];
                b_vals[0] = b[(l + 0) * n + j];
                b_vals[1] = b[(l + 1) * n + j];
                float64x2_t b_vec = vld1q_f64(b_vals);

                // Fused multiply-add: sum_vec += a_vec * b_vec
                sum_vec = vfmaq_f64(sum_vec, a_vec, b_vec);
            }

            // Horizontal reduction: sum all elements in sum_vec
            double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

            // Handle remaining elements
            for (; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }

            c[i * n + j] = sum;
        }
    }
}

// matmul_neon_optimized: Cache-blocked version for better performance
void matmul_neon_optimized(double* restrict c, const double* restrict a,
                           const double* restrict b, int64_t m, int64_t n, int64_t k) {

    const int64_t NEON_WIDTH = 2;
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
                        float64x2_t sum_vec = vdupq_n_f64(0.0);

                        int64_t l = kk;

                        // Vectorized inner loop
                        for (; l + NEON_WIDTH <= k_max; l += NEON_WIDTH) {
                            float64x2_t a_vec = vld1q_f64(&a[i * k + l]);

                            double b_vals[2];
                            b_vals[0] = b[(l + 0) * n + j];
                            b_vals[1] = b[(l + 1) * n + j];
                            float64x2_t b_vec = vld1q_f64(b_vals);

                            sum_vec = vfmaq_f64(sum_vec, a_vec, b_vec);
                        }

                        c[i * n + j] += vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

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

// Query NEON availability (always available on ARM64)
int neon_available(void) {
    return 1;  // NEON is mandatory on ARM64
}
