#pragma once
#include "kernel_common.cuh"
#include <cublas_v2.h>

/*
 * Kernel 0 / 8 – cuBLAS Reference
 *
 * Wraps cuBLAS SGEMM as the performance baseline.
 * cuBLAS uses column-major by default, but our matrices are row-major.
 * Trick: C = A * B  in row-major  ⟺  C^T = B^T * A^T  in column-major
 *        Since (AB)^T = B^T A^T, we call cublasSgemm with swapped A/B.
 */

static cublasHandle_t cublas_handle = nullptr;

inline void cublas_init() {
  if (cublas_handle == nullptr) {
    cublasCreate(&cublas_handle);
  }
}

inline void cublas_destroy() {
  if (cublas_handle != nullptr) {
    cublasDestroy(cublas_handle);
    cublas_handle = nullptr;
  }
}

inline void launch_kernel_cublas(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  cublas_init();
  // Row-major trick: interpret as column-major with transposed dimensions
  // C(M×N) = alpha * A(M×K) * B(K×N) + beta * C(M×N)   [row-major]
  // ≡ C^T(N×M) = alpha * B^T(N×K) * A^T(K×M) + beta * C^T(N×M)  [col-major]
  cublasSgemm(cublas_handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K,
              &alpha,
              B, N,    // B^T in col-major = B in row-major, leading dim = N
              A, K,    // A^T in col-major = A in row-major, leading dim = K
              &beta,
              C, N);   // C^T in col-major = C in row-major, leading dim = N
}
