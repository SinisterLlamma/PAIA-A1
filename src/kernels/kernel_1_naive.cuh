#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 1 – Naive matrix multiplication
 *
 * Each thread computes exactly one element of C.
 * Grid:  (ceil(M/32), ceil(N/32))   — 2D grid of blocks
 * Block: (32, 32)                   — 1024 threads per block
 *
 * Memory pattern: threads in the same warp (consecutive threadIdx.x) access
 * different *rows* of A but the *same column* of B.  Because A is row-major,
 * adjacent threads hit addresses that are K floats apart → non-coalesced.
 */

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // Each thread's output position in C
  const int row = blockIdx.x * blockDim.x + threadIdx.x; // row of C
  const int col = blockIdx.y * blockDim.y + threadIdx.y; // col of C

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

inline void launch_kernel_naive(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta,
                                float *C) {
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
