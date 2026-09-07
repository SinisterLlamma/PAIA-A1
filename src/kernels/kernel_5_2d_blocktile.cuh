#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 5 – 2D Block-Tiling
 *
 * Each thread computes a TM×TN sub-tile of C. This dramatically increases the
 * compute-to-memory ratio: each loaded element from SMEM is reused across
 * both the TM and TN dimensions → O(TM*TN) compute per O(TM+TN) loads.
 *
 * Block tile:  BM × BN (in SMEM)
 * Thread tile: TM × TN (in registers)
 * Threads per block: (BM/TM) * (BN/TN) = 256
 *
 * Uses blockIdx.y for M and blockIdx.x for N for optimal L2 cache locality.
 */

template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void sgemm_2d_blocktile(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int totalThreads = (BM / TM) * (BN / TN);
  const int tid = threadIdx.x;

  const int threadCol = tid % (BN / TN);
  const int threadRow = tid / (BN / TN);

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM * TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  for (int tileK = 0; tileK < K; tileK += BK) {
    // Collaborative load of A (BM × BK)
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

    // Collaborative load of B (BK × BN)
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

    // Compute TM×TN outer products per k step
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int tm = 0; tm < TM; ++tm) {
        regA[tm] = As[(threadRow * TM + tm) * BK + dotIdx];
      }
      for (int tn = 0; tn < TN; ++tn) {
        regB[tn] = Bs[dotIdx * BN + threadCol * TN + tn];
      }
      for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
          threadResults[tm * TN + tn] += regA[tm] * regB[tn];
        }
      }
    }

    __syncthreads();

    A += BK;
    B += BK * N;
  }

  // Write results to C
  for (int tm = 0; tm < TM; ++tm) {
    for (int tn = 0; tn < TN; ++tn) {
      int gRow = cRow * BM + threadRow * TM + tm;
      int gCol = cCol * BN + threadCol * TN + tn;
      if (gRow < M && gCol < N) {
        C[(threadRow * TM + tm) * N + threadCol * TN + tn] =
            alpha * threadResults[tm * TN + tn] +
            beta * C[(threadRow * TM + tm) * N + threadCol * TN + tn];
      }
    }
  }
}

inline void launch_kernel_2d_blocktile(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  if (M >= 128 && N >= 128) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_2d_blocktile<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    const int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_2d_blocktile<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  }
}
