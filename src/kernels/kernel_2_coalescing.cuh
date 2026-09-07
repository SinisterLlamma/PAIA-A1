#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 2 – Global Memory Coalescing
 *
 * Same computation as Kernel 1, but we use a 1D blockDim and manually
 * compute (row, col) so that threads with consecutive threadIdx.x map to
 * consecutive *columns* of C.  Since C and B are row-major, this means
 * consecutive threads access consecutive memory addresses → coalesced.
 *
 * Block: (BLOCKSIZE * BLOCKSIZE, 1, 1)  — 1D
 * Grid:  (ceil(M/BLOCKSIZE), ceil(N/BLOCKSIZE))
 */

template <int BLOCKSIZE = 32>
__global__ void sgemm_coalescing(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  // Remap flat threadIdx.x → 2D (row, col) so that consecutive threads
  // get consecutive columns (col increments fastest)
  const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

inline void launch_kernel_coalescing(int M, int N, int K, float alpha,
                                     const float *A, const float *B,
                                     float beta, float *C) {
  const int BS = 32;
  dim3 block(BS * BS);
  dim3 grid(CEIL_DIV(M, BS), CEIL_DIV(N, BS));
  sgemm_coalescing<BS><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
