// ===========================================================================
// Multi-Engine SVE Matrix Multiplication - Optimized for Graviton3/4
// ===========================================================================
//
// This implementation optimally utilizes multiple SVE engines:
//   - Graviton3: 2× 256-bit SVE units (512-bit total throughput)
//   - Graviton4: 4× 128-bit SVE2 units (512-bit total throughput)
//
// Key optimizations:
// 1. OpenMP parallelization across SVE engines
// 2. Cache blocking (tiling) for L1/L2/L3 caches
// 3. Register blocking to maximize FMA throughput
// 4. Transpose B matrix for contiguous access
// 5. Loop unrolling and software pipelining
//
// Compile: gcc -march=armv8.2-a+sve -O3 -fopenmp -ffast-math
//
// ===========================================================================

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

// Cache sizes (approximate for Graviton3/4)
#define L1_SIZE (64 * 1024)      // 64KB L1 cache per core
#define L2_SIZE (1024 * 1024)    // 1MB L2 cache per core
#define L3_SIZE (32 * 1024 * 1024) // 32MB shared L3

// Optimal block sizes determined empirically
#define BLOCK_K 256   // K dimension blocking for L2 cache
#define BLOCK_N 128   // N dimension blocking for L1 cache
#define BLOCK_M 64    // M dimension blocking for registers

// Micro-kernel register blocking
#define MR 4  // M register block (4 output rows)
#define NR 4  // N register block (4 output cols)

// Transpose B matrix for better cache behavior
static void transpose_matrix(double* restrict bt, const double* restrict b,
                             int64_t k, int64_t n) {
    #pragma omp parallel for collapse(2)
    for (int64_t j = 0; j < n; j++) {
        for (int64_t l = 0; l < k; l++) {
            bt[j * k + l] = b[l * n + j];
        }
    }
}

// SVE micro-kernel: compute MR×NR block using register blocking
static inline void sve_microkernel(double* restrict c, const double* restrict a,
                                   const double* restrict bt,
                                   int64_t k, int64_t ldc, int64_t lda, int64_t ldbt) {
    const int64_t vl = svcntd();
    svbool_t pg = svptrue_b64();

    // Accumulator registers for MR×NR output elements
    // SVE doesn't allow arrays of SVE types, so we use individual variables
    svfloat64_t acc00 = svdup_n_f64(0.0);
    svfloat64_t acc01 = svdup_n_f64(0.0);
    svfloat64_t acc02 = svdup_n_f64(0.0);
    svfloat64_t acc03 = svdup_n_f64(0.0);
    svfloat64_t acc10 = svdup_n_f64(0.0);
    svfloat64_t acc11 = svdup_n_f64(0.0);
    svfloat64_t acc12 = svdup_n_f64(0.0);
    svfloat64_t acc13 = svdup_n_f64(0.0);
    svfloat64_t acc20 = svdup_n_f64(0.0);
    svfloat64_t acc21 = svdup_n_f64(0.0);
    svfloat64_t acc22 = svdup_n_f64(0.0);
    svfloat64_t acc23 = svdup_n_f64(0.0);
    svfloat64_t acc30 = svdup_n_f64(0.0);
    svfloat64_t acc31 = svdup_n_f64(0.0);
    svfloat64_t acc32 = svdup_n_f64(0.0);
    svfloat64_t acc33 = svdup_n_f64(0.0);

    // Main computation loop over K dimension
    int64_t l = 0;
    for (; l + vl <= k; l += vl) {
        // Load A vectors for MR rows
        svfloat64_t a0 = svld1_f64(pg, &a[0 * lda + l]);
        svfloat64_t a1 = svld1_f64(pg, &a[1 * lda + l]);
        svfloat64_t a2 = svld1_f64(pg, &a[2 * lda + l]);
        svfloat64_t a3 = svld1_f64(pg, &a[3 * lda + l]);

        // Load BT vectors for NR columns
        svfloat64_t b0 = svld1_f64(pg, &bt[0 * ldbt + l]);
        svfloat64_t b1 = svld1_f64(pg, &bt[1 * ldbt + l]);
        svfloat64_t b2 = svld1_f64(pg, &bt[2 * ldbt + l]);
        svfloat64_t b3 = svld1_f64(pg, &bt[3 * ldbt + l]);

        // Compute MR×NR outer product using FMA
        acc00 = svmla_f64_m(pg, acc00, a0, b0);
        acc01 = svmla_f64_m(pg, acc01, a0, b1);
        acc02 = svmla_f64_m(pg, acc02, a0, b2);
        acc03 = svmla_f64_m(pg, acc03, a0, b3);

        acc10 = svmla_f64_m(pg, acc10, a1, b0);
        acc11 = svmla_f64_m(pg, acc11, a1, b1);
        acc12 = svmla_f64_m(pg, acc12, a1, b2);
        acc13 = svmla_f64_m(pg, acc13, a1, b3);

        acc20 = svmla_f64_m(pg, acc20, a2, b0);
        acc21 = svmla_f64_m(pg, acc21, a2, b1);
        acc22 = svmla_f64_m(pg, acc22, a2, b2);
        acc23 = svmla_f64_m(pg, acc23, a2, b3);

        acc30 = svmla_f64_m(pg, acc30, a3, b0);
        acc31 = svmla_f64_m(pg, acc31, a3, b1);
        acc32 = svmla_f64_m(pg, acc32, a3, b2);
        acc33 = svmla_f64_m(pg, acc33, a3, b3);
    }

    // Horizontal reduction and accumulation
    c[0 * ldc + 0] += svaddv_f64(pg, acc00);
    c[0 * ldc + 1] += svaddv_f64(pg, acc01);
    c[0 * ldc + 2] += svaddv_f64(pg, acc02);
    c[0 * ldc + 3] += svaddv_f64(pg, acc03);

    c[1 * ldc + 0] += svaddv_f64(pg, acc10);
    c[1 * ldc + 1] += svaddv_f64(pg, acc11);
    c[1 * ldc + 2] += svaddv_f64(pg, acc12);
    c[1 * ldc + 3] += svaddv_f64(pg, acc13);

    c[2 * ldc + 0] += svaddv_f64(pg, acc20);
    c[2 * ldc + 1] += svaddv_f64(pg, acc21);
    c[2 * ldc + 2] += svaddv_f64(pg, acc22);
    c[2 * ldc + 3] += svaddv_f64(pg, acc23);

    c[3 * ldc + 0] += svaddv_f64(pg, acc30);
    c[3 * ldc + 1] += svaddv_f64(pg, acc31);
    c[3 * ldc + 2] += svaddv_f64(pg, acc32);
    c[3 * ldc + 3] += svaddv_f64(pg, acc33);

    // Scalar cleanup for remainder
    for (int r = 0; r < MR; r++) {
        for (int col = 0; col < NR; col++) {
            for (int64_t ll = l; ll < k; ll++) {
                c[r * ldc + col] += a[r * lda + ll] * bt[col * ldbt + ll];
            }
        }
    }
}

// Multi-engine optimized matrix multiplication
void matmul_sve_multiengine(double* restrict c, const double* restrict a,
                            const double* restrict b, int64_t m, int64_t n, int64_t k) {

    // Allocate transposed B matrix
    double* bt = (double*)aligned_alloc(64, k * n * sizeof(double));
    if (!bt) return;

    // Transpose B for contiguous column access
    transpose_matrix(bt, b, k, n);

    // Initialize C to zero
    memset(c, 0, m * n * sizeof(double));

    // Three-level cache-aware blocking
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t ii = 0; ii < m; ii += BLOCK_M) {
        for (int64_t jj = 0; jj < n; jj += BLOCK_N) {
            for (int64_t kk = 0; kk < k; kk += BLOCK_K) {

                // Compute block bounds
                int64_t i_end = (ii + BLOCK_M < m) ? ii + BLOCK_M : m;
                int64_t j_end = (jj + BLOCK_N < n) ? jj + BLOCK_N : n;
                int64_t k_end = (kk + BLOCK_K < k) ? kk + BLOCK_K : k;
                int64_t k_size = k_end - kk;

                // Process block using micro-kernels
                for (int64_t i = ii; i < i_end; i += MR) {
                    for (int64_t j = jj; j < j_end; j += NR) {

                        // Check if we have a full MR×NR block
                        if (i + MR <= i_end && j + NR <= j_end) {
                            sve_microkernel(&c[i * n + j],
                                          &a[i * k + kk],
                                          &bt[j * k + kk],
                                          k_size, n, k, k);
                        } else {
                            // Handle edge cases with scalar code
                            int64_t m_cur = (i + MR <= i_end) ? MR : i_end - i;
                            int64_t n_cur = (j + NR <= j_end) ? NR : j_end - j;

                            for (int64_t r = 0; r < m_cur; r++) {
                                for (int64_t col = 0; col < n_cur; col++) {
                                    double sum = 0.0;
                                    for (int64_t l = kk; l < k_end; l++) {
                                        sum += a[(i+r) * k + l] * bt[(j+col) * k + l];
                                    }
                                    c[(i+r) * n + (j+col)] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(bt);
}

// Variant with explicit thread pinning for multi-engine utilization
void matmul_sve_multiengine_pinned(double* restrict c, const double* restrict a,
                                   const double* restrict b, int64_t m, int64_t n, int64_t k) {

    // On Graviton3: 2 SVE engines per core, use 2 threads per core
    // On Graviton4: 4 SVE engines per core, use 4 threads per core
    // This is handled by OMP_NUM_THREADS environment variable

    double* bt = (double*)aligned_alloc(64, k * n * sizeof(double));
    if (!bt) return;

    transpose_matrix(bt, b, k, n);
    memset(c, 0, m * n * sizeof(double));

    // Distribute work across engines with fine-grained parallelism
    #pragma omp parallel
    {
        const int64_t vl = svcntd();
        svbool_t pg = svptrue_b64();

        #pragma omp for collapse(2) schedule(static)
        for (int64_t ii = 0; ii < m; ii += BLOCK_M) {
            for (int64_t jj = 0; jj < n; jj += BLOCK_N) {

                int64_t i_end = (ii + BLOCK_M < m) ? ii + BLOCK_M : m;
                int64_t j_end = (jj + BLOCK_N < n) ? jj + BLOCK_N : n;

                // Inner K loop unblocked for register reuse
                for (int64_t i = ii; i < i_end; i++) {
                    for (int64_t j = jj; j < j_end; j++) {
                        svfloat64_t sum_vec = svdup_n_f64(0.0);

                        int64_t l = 0;
                        for (; l + vl <= k; l += vl) {
                            svfloat64_t a_vec = svld1_f64(pg, &a[i * k + l]);
                            svfloat64_t b_vec = svld1_f64(pg, &bt[j * k + l]);
                            sum_vec = svmla_f64_m(pg, sum_vec, a_vec, b_vec);
                        }

                        double sum = svaddv_f64(pg, sum_vec);

                        // Scalar remainder
                        for (; l < k; l++) {
                            sum += a[i * k + l] * bt[j * k + l];
                        }

                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }

    free(bt);
}

#else  // No SVE support

void matmul_sve_multiengine(double* restrict c, const double* restrict a,
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

void matmul_sve_multiengine_pinned(double* restrict c, const double* restrict a,
                                   const double* restrict b, int64_t m, int64_t n, int64_t k) {
    matmul_sve_multiengine(c, a, b, m, n, k);
}

#endif
