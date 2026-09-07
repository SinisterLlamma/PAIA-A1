#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 7 – Resolving Shared Memory Bank Conflicts via Padding
 *
 * Directly based on Simon Boehm's Kernel 8 (8_kernel_bank_extra_col.cuh):
 * Shared memory consists of 32 banks of 4-byte width. When multiple threads in
 * a warp access different words belonging to the same memory bank, the accesses
 * are serialized (bank conflict).
 *
 * In Kernel 6, Bs has stride BN=128 floats per row. Because 128 is a multiple of 32,
 * successive rows align to the exact same banks. By adding padding (extraCols = 5):
 *   __shared__ float Bs[BK * (BN + 5)];
 * The stride becomes 133 floats, which is coprime to 32. This breaks the 32-way
 * bank alignment and eliminates shared memory bank conflicts on Bs!
 */

template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8, int extraCols = 5>
__global__ void sgemm_bank_padding_fast(int M, int N, int K, float alpha,
                                        const float *A, const float *B, float beta,
                                        float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * (BN + extraCols)];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Transpose A while loading into SMEM
    float4 tmpA = reinterpret_cast<const float4 *>(
        &A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmpA.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmpA.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmpA.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmpA.w;

    // Load B into padded shared memory to eliminate bank conflicts
    float4 tmpB = reinterpret_cast<const float4 *>(
        &B[innerRowB * N + innerColB * 4])[0];
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmpB.x;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmpB.y;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmpB.z;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmpB.w;

    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
      }
      for (uint resM = 0; resM < TM; ++resM) {
        for (uint resN = 0; resN < TN; ++resN) {
          threadResults[resM * TN + resN] += regM[resM] * regN[resN];
        }
      }
    }

    __syncthreads();
  }

  // Vectorized write-back to C
  for (uint resM = 0; resM < TM; ++resM) {
    for (uint resN = 0; resN < TN; resN += 4) {
      float4 tmpC = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0];
      tmpC.x = alpha * threadResults[resM * TN + resN + 0] + beta * tmpC.x;
      tmpC.y = alpha * threadResults[resM * TN + resN + 1] + beta * tmpC.y;
      tmpC.z = alpha * threadResults[resM * TN + resN + 2] + beta * tmpC.z;
      tmpC.w = alpha * threadResults[resM * TN + resN + 3] + beta * tmpC.w;
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0] = tmpC;
    }
  }
}

// Safe fallback for arbitrary/odd matrices
template <int BM = 64, int BN = 64, int BK = 8, int TM = 4, int TN = 4, int extraCols = 5>
__global__ void sgemm_bank_padding_safe(int M, int N, int K, float alpha,
                                        const float *A, const float *B, float beta,
                                        float *C) {
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * (BN + extraCols)];

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
    for (int offset = 0; offset < BM * BK; offset += totalThreads) {
      int idx = offset + tid;
      if (idx < BM * BK) {
        int r = idx / BK;
        int c = idx % BK;
        if (cRow * BM + r < M && tileK + c < K)
          As[c * BM + r] = A[r * K + c];
        else
          As[c * BM + r] = 0.0f;
      }
    }

    for (int offset = 0; offset < BK * BN; offset += totalThreads) {
      int idx = offset + tid;
      if (idx < BK * BN) {
        int r = idx / BN;
        int c = idx % BN;
        if (tileK + r < K && cCol * BN + c < N)
          Bs[r * (BN + extraCols) + c] = B[r * N + c];
        else
          Bs[r * (BN + extraCols) + c] = 0.0f;
      }
    }

    __syncthreads();

    for (int k = 0; k < BK; ++k) {
      for (int tm = 0; tm < TM; ++tm) {
        regA[tm] = As[k * BM + threadRow * TM + tm];
      }
      for (int tn = 0; tn < TN; ++tn) {
        regB[tn] = Bs[k * (BN + extraCols) + threadCol * TN + tn];
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

inline void launch_kernel_bank_conflicts(int M, int N, int K, float alpha,
                                         const float *A, const float *B,
                                         float beta, float *C) {
  if (M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
      ((uintptr_t)A % 16 == 0) && ((uintptr_t)B % 16 == 0) && ((uintptr_t)C % 16 == 0)) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_bank_padding_fast<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    const int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_bank_padding_safe<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  }
}
