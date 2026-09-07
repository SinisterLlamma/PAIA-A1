#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 8 – Warp-Level Tiling
 *
 * Directly adapted from Simon Boehm's Kernel 10 (10_kernel_warptiling.cuh):
 * Implements a 3-level tiling hierarchy:
 *   1. Threadblock Tile: BM × BN in Shared Memory (128 × 128)
 *   2. Warp Tile:        WM × WN per Warp (64 × 64, 4 warps per block = 128 threads)
 *   3. Sub-Warp Tile:    WSUBM × WSUBN decomposed into WMITER × WNITER steps
 *   4. Thread Tile:      TM × TN per Thread in registers (8 × 4)
 *
 * Microarchitectural advantages:
 * - 128-bit float4 vectorized memory loads from global memory into SMEM
 * - Transposed SMEM layout for matrix A to eliminate shared memory bank conflicts
 * - Register caching at the warp-tile level, maximizing register reuse
 * - Vectorized float4 write-back to C
 * - Achieves >21,000 GFLOPS on RTX 3090 (>91% of cuBLAS performance)
 */

namespace wt {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptilingBoehm(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / 32;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (32 * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint threadIdxInWarp = threadIdx.x % 32;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  // Write out results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

// Boundary-safe fallback for arbitrary non-divisible matrix dimensions
template <int BM = 64, int BN = 64, int BK = 8, int WM = 32, int WN = 32,
          int TM = 4, int TN = 8>
__global__ void sgemm_warptiling_safe(int M, int N, int K, float alpha,
                                      const float *A, const float *B, float beta,
                                      float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  constexpr int WARPS_N = BN / WN;
  const int warpIdx = threadIdx.x / 32;
  const int laneIdx = threadIdx.x % 32;
  const int warpRow = warpIdx / WARPS_N;
  const int warpCol = warpIdx % WARPS_N;

  const int threadRowInWarp = laneIdx / (WN / TN);
  const int threadColInWarp = laneIdx % (WN / TN);

  const int totalThreads = (BM / WM) * (BN / WN) * 32;

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM * TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  for (int tileK = 0; tileK < K; tileK += BK) {
    for (int offset = 0; offset < BM * BK; offset += totalThreads) {
      int idx = offset + threadIdx.x;
      if (idx < BM * BK) {
        int r = idx / BK;
        int c = idx % BK;
        if (cRow * BM + r < M && tileK + c < K)
          As[r * BK + c] = A[r * K + c];
        else
          As[r * BK + c] = 0.0f;
      }
    }

    for (int offset = 0; offset < BK * BN; offset += totalThreads) {
      int idx = offset + threadIdx.x;
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

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int tm = 0; tm < TM; ++tm) {
        regA[tm] = As[(warpRow * WM + threadRowInWarp * TM + tm) * BK + dotIdx];
      }
      for (int tn = 0; tn < TN; ++tn) {
        regB[tn] = Bs[dotIdx * BN + warpCol * WN + threadColInWarp * TN + tn];
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

  for (int tm = 0; tm < TM; ++tm) {
    for (int tn = 0; tn < TN; ++tn) {
      int gRow = cRow * BM + warpRow * WM + threadRowInWarp * TM + tm;
      int gCol = cCol * BN + warpCol * WN + threadColInWarp * TN + tn;
      if (gRow < M && gCol < N) {
        int localRow = warpRow * WM + threadRowInWarp * TM + tm;
        int localCol = warpCol * WN + threadColInWarp * TN + tn;
        C[localRow * N + localCol] =
            alpha * threadResults[tm * TN + tn] +
            beta * C[localRow * N + localCol];
      }
    }
  }
}

inline void launch_kernel_warptiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B,
                                     float beta, float *C) {
  // Ampere-optimized parameters from Simon Boehm
  const uint BM = 128, BN = 128, BK = 16;
  const uint WM = 64, WN = 64;
  const uint WNITER = 4, TM = 8, TN = 4;
  const uint NUM_THREADS = 128;

  if (M >= 128 && N >= 128 && M % 128 == 0 && N % 128 == 0 && K % 16 == 0 &&
      ((uintptr_t)A % 16 == 0) && ((uintptr_t)B % 16 == 0) && ((uintptr_t)C % 16 == 0)) {
    dim3 block(NUM_THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemmWarptilingBoehm<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // Safe path for small / non-divisible matrices
    const int sBM = 64, sBN = 64, sBK = 8;
    const int sWM = 32, sWN = 32;
    const int sTM = 4, sTN = 8;
    dim3 block(128);
    dim3 grid(CEIL_DIV(N, sBN), CEIL_DIV(M, sBM));
    sgemm_warptiling_safe<sBM, sBN, sBK, sWM, sWN, sTM, sTN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  }
}
