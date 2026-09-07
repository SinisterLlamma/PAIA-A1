#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 4 – 1D Block-Tiling (Thread Coarsening)
 *
 * Instead of each thread computing a single element of C, each thread
 * computes TM elements arranged as a column vector. This increases the
 * compute-to-memory ratio: each loaded B element is reused TM times in
 * registers.
 *
 * Block tile:  BM × BN (from SMEM)
 * Thread tile: TM × 1  (each thread computes TM elements of C)
 * Threads per block: (BM * BN) / TM
 *
 * Crucial insight from Simon Boehm:
 * Using blockIdx.y for rows and blockIdx.x for columns ensures that blocks
 * with sequential blockIDs access columns of B sequentially while sharing
 * rows of A. This yields significantly better spatial locality in L2 cache (~30% speedup).
 */

template <int BM = 64, int BN = 64, int BK = 8, int TM = 8>
__global__ void sgemm_1d_blocktile(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Simon Boehm mapping: x -> columns, y -> rows for better L2 spatial locality
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int totalThreads = (BM * BN) / TM;
  const int tid = threadIdx.x;

  const int threadCol = tid % BN;
  const int threadRow = tid / BN;

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM] = {0.0f};

  for (int tileK = 0; tileK < K; tileK += BK) {
    // Collaborative load A
    for (int offset = 0; offset < BM * BK; offset += totalThreads) {
      int idx = offset + tid;
      if (idx < BM * BK) {
        int r = idx / BK;
        int c = idx % BK;
        if (cRow * BM + r < M && tileK + c < K)
          As[r * BK + c] = A[r * K + c];
        else
          As[r * BK + c] = 0.0f;
      }
    }

    // Collaborative load B
    for (int offset = 0; offset < BK * BN; offset += totalThreads) {
      int idx = offset + tid;
      if (idx < BK * BN) {
        int r = idx / BN;
        int c = idx % BN;
        if (tileK + r < K && cCol * BN + c < N)
          Bs[r * BN + c] = B[r * N + c];
        else
          Bs[r * BN + c] = 0.0f;
      }
    }

    __syncthreads();

    // Compute per-thread results: dot product across BK
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }

    __syncthreads();

    A += BK;
    B += BK * N;
  }

  // Write results to C
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    int gRow = cRow * BM + threadRow * TM + resIdx;
    int gCol = cCol * BN + threadCol;
    if (gRow < M && gCol < N) {
      C[(threadRow * TM + resIdx) * N + threadCol] =
          alpha * threadResults[resIdx] +
          beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
  }
}

inline void launch_kernel_1d_blocktile(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;
  dim3 block((BM * BN) / TM);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_1d_blocktile<BM, BN, BK, TM>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
