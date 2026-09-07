#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 7 – Warp Tiling
 *
 * Three-level tiling hierarchy: Block → Warp → Thread
 *
 * Block tile:  BM × BN (loaded into shared memory)
 * Warp tile:   WM × WN (a sub-region of the block tile)
 * Thread tile: TM × TN (each thread computes this many results)
 *
 * Within each warp:
 *   - The warp handles a WM × WN region of the block tile
 *   - Threads are mapped as a 2D grid: (WM/TM) × (WN/TN) = 32 threads
 *   - Each thread computes a TM × TN sub-tile using register-cached fragments
 *
 * The key benefit over plain 2D blocktiling is better locality:
 * threads within the same warp share SMEM reads more effectively
 * since they're mapped to nearby rows/columns.
 */

template <int BM = 64, int BN = 64, int BK = 8, int WM = 32, int WN = 32,
          int TM = 4, int TN = 4>
__global__ void sgemm_warptiling(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int bRow = blockIdx.x;
  const int bCol = blockIdx.y;

  // Warp-level indexing
  constexpr int WARPS_M = BM / WM;     // warps per block in M dim
  constexpr int WARPS_N = BN / WN;     // warps per block in N dim
  const int warpIdx = threadIdx.x / 32;
  const int laneIdx = threadIdx.x % 32;
  const int warpRow = warpIdx / WARPS_N;  // this warp's row in the block tile
  const int warpCol = warpIdx % WARPS_N;  // this warp's col in the block tile

  // Thread-level indexing within the warp tile
  // WM/TM threads in M direction, WN/TN threads in N direction
  // (WM/TM) * (WN/TN) must equal 32 (warp size)
  constexpr int THREADS_M = WM / TM;  // = 8
  constexpr int THREADS_N = WN / TN;  // = 8, so 8*4 threads... wait
  // With WM=32,TM=4: THREADS_M=8; WN=32,TN=4: THREADS_N=8; 8*8=64 ≠ 32
  // So we need: (WM/TM)*(WN/TN) = 32 → with WM=WN=32: TM*TN = 32
  // E.g., TM=4, TN=4 → 8*8=64... nope. Need TM=8, TN=4 → 4*8=32 ✓
  // Or TM=4, TN=8 → 8*4=32 ✓
  // Let's just compute based on actual params and let threads handle
  // multiple sub-tiles if needed.

  // Simple approach: each thread in the warp maps to a position in the
  // warp tile. If WM/TM * WN/TN > 32, each thread loops.
  // If WM/TM * WN/TN <= 32, some threads are idle (but we size to avoid this).
  const int threadRowInWarp = laneIdx / (WN / TN);  // [0, WM/TM)
  const int threadColInWarp = laneIdx % (WN / TN);  // [0, WN/TN)

  const int totalThreads = WARPS_M * WARPS_N * 32;

  // Advance pointers
  A += bRow * BM * K;
  B += bCol * BN;
  C += bRow * BM * N + bCol * BN;

  // Result registers
  float threadResults[TM * TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  for (int tileK = 0; tileK < K; tileK += BK) {
    // ---- Collaborative load A and B tiles into SMEM ----
    for (int loadOff = 0; loadOff < BM * BK; loadOff += totalThreads) {
      int idx = loadOff + threadIdx.x;
      if (idx < BM * BK) {
        int r = idx / BK;
        int c = idx % BK;
        if (bRow * BM + r < M && tileK + c < K)
          As[r * BK + c] = A[r * K + c];
        else
          As[r * BK + c] = 0.0f;
      }
    }

    for (int loadOff = 0; loadOff < BK * BN; loadOff += totalThreads) {
      int idx = loadOff + threadIdx.x;
      if (idx < BK * BN) {
        int r = idx / BN;
        int c = idx % BN;
        if (tileK + r < K && bCol * BN + c < N)
          Bs[r * BN + c] = B[r * N + c];
        else
          Bs[r * BN + c] = 0.0f;
      }
    }

    __syncthreads();

    // ---- Compute: warp-tiled ----
    for (int k = 0; k < BK; ++k) {
      // Load A fragment: TM elements from this thread's row range
      for (int tm = 0; tm < TM; ++tm) {
        int aRow = warpRow * WM + threadRowInWarp * TM + tm;
        regA[tm] = (aRow < BM) ? As[aRow * BK + k] : 0.0f;
      }
      // Load B fragment: TN elements from this thread's col range
      for (int tn = 0; tn < TN; ++tn) {
        int bColIdx = warpCol * WN + threadColInWarp * TN + tn;
        regB[tn] = (bColIdx < BN) ? Bs[k * BN + bColIdx] : 0.0f;
      }
      // Outer product
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

  // ---- Write results ----
  for (int tm = 0; tm < TM; ++tm) {
    for (int tn = 0; tn < TN; ++tn) {
      int globalRow = bRow * BM + warpRow * WM + threadRowInWarp * TM + tm;
      int globalCol = bCol * BN + warpCol * WN + threadColInWarp * TN + tn;
      if (globalRow < M && globalCol < N) {
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
  // Parameters chosen so that (WM/TM)*(WN/TN) = warp size = 32
  // WM=32, TM=4 → 8 threads in M
  // WN=32, TN=8 → 4 threads in N
  // 8 * 4 = 32 ✓
  const int BM = 64, BN = 64, BK = 8;
  const int WM = 32, WN = 32;
  const int TM = 4, TN = 8;

  const int WARPS_M = BM / WM;  // 2
  const int WARPS_N = BN / WN;  // 2
  const int numWarps = WARPS_M * WARPS_N; // 4
  const int threadsPerBlock = numWarps * 32; // 128

  dim3 block(threadsPerBlock);
  dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
  sgemm_warptiling<BM, BN, BK, WM, WN, TM, TN>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
