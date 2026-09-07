#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 3 – Shared Memory Cache-Blocking (Tiled GEMM)
 *
 * Each block loads a BLOCKSIZE×BLOCKSIZE tile of A and B into shared memory,
 * then all threads in the block compute partial dot products from SMEM.
 * We slide the tiles across the K dimension, accumulating into registers.
 *
 * This reduces global memory traffic by a factor of ~BLOCKSIZE compared
 * to Kernel 2, since each element loaded into SMEM is reused BLOCKSIZE times.
 *
 * Block: (BLOCKSIZE * BLOCKSIZE, 1, 1)  — 1D for coalesced loads
 * Grid:  (ceil(M/BLOCKSIZE), ceil(N/BLOCKSIZE))
 */

template <int BLOCKSIZE = 32>
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  // Block's starting row/col in C
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  // Thread's position within the block
  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;

  // Advance pointers to this block's starting positions
  A += cRow * BLOCKSIZE * K;           // row = cRow*BS, col = 0
  B += cCol * BLOCKSIZE;               // row = 0, col = cCol*BS
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // (cRow*BS, cCol*BS)

  float acc = 0.0f;

  // Slide the tile along K
  for (int tileIdx = 0; tileIdx < K; tileIdx += BLOCKSIZE) {
    // Collaborative load: each thread loads one element of A and B
    // threadCol is the fast-varying index → coalesced access
    if (cRow * BLOCKSIZE + threadRow < M && tileIdx + threadCol < K)
      As[threadRow][threadCol] = A[threadRow * K + threadCol];
    else
      As[threadRow][threadCol] = 0.0f;

    if (tileIdx + threadRow < K && cCol * BLOCKSIZE + threadCol < N)
      Bs[threadRow][threadCol] = B[threadRow * N + threadCol];
    else
      Bs[threadRow][threadCol] = 0.0f;

    __syncthreads();

    // Compute partial dot product from this tile
    for (int k = 0; k < BLOCKSIZE; ++k) {
      acc += As[threadRow][k] * Bs[k][threadCol];
    }

    __syncthreads();

    // Advance tile
    A += BLOCKSIZE;       // move right along A's columns
    B += BLOCKSIZE * N;   // move down along B's rows
  }

  // Write result
  if (cRow * BLOCKSIZE + threadRow < M && cCol * BLOCKSIZE + threadCol < N) {
    C[threadRow * N + threadCol] = alpha * acc + beta * C[threadRow * N + threadCol];
  }
}

inline void launch_kernel_smem(int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C) {
  const int BS = 32;
  dim3 block(BS * BS);
  dim3 grid(CEIL_DIV(M, BS), CEIL_DIV(N, BS));
  sgemm_smem<BS><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
