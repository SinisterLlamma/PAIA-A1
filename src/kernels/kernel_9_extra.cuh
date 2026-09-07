#pragma once
#include "kernel_common.cuh"

/*
 * Kernel 9 – Extra / Alternative Algorithms
 *
 * Two alternative approaches for comparison:
 *
 * (a) Transpose-then-Multiply: Pre-transpose B so that both A and B^T
 *     are accessed row-wise, improving cache behavior even in the naive case.
 *
 * (b) Recursive tiling: A simple demonstration of recursive cache-oblivious
 *     decomposition on GPU. Not expected to beat hand-tuned tiling but
 *     interesting for analysis.
 */

// ===== (a) Transpose B kernel =====
__global__ void transpose_kernel(const float *in, float *out, int rows,
                                 int cols) {
  // Transpose rows×cols matrix → cols×rows
  __shared__ float tile[32][33]; // +1 to avoid bank conflicts

  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;

  if (x < cols && y < rows) {
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  }
  __syncthreads();

  // Transposed coordinates
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;

  if (x < rows && y < cols) {
    out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
  }
}

// Matmul with transposed B: C = alpha * A * B^T^T + beta * C
// where Bt is already B transposed (N×K)
// So A(M×K) * Bt^T(K×N) = A(M×K) dot-product rows of A with rows of Bt
__global__ void sgemm_with_transpose(int M, int N, int K, float alpha,
                                     const float *A, const float *Bt,
                                     float beta, float *C) {
  // Bt is N×K (transposed B), so Bt[col][k] = B[k][col]
  // C[row][col] = sum_k A[row][k] * B[k][col] = sum_k A[row][k] * Bt[col][k]
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * Bt[col * K + k];
      // Both accesses stride by 1 for consecutive k → better cache
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ===== (b) Recursive Tiled Matmul =====
// GPU-side recursive tiling: each block handles a tile using SMEM,
// with double-buffering for overlapping loads and compute.
template <int TILE = 32>
__global__ void sgemm_recursive_tile(int M, int N, int K, float alpha,
                                     const float *A, const float *B,
                                     float beta, float *C) {
  // Double-buffered shared memory
  __shared__ float As[2][TILE][TILE];
  __shared__ float Bs[2][TILE][TILE];

  const int bRow = blockIdx.x;
  const int bCol = blockIdx.y;
  const int tRow = threadIdx.y;
  const int tCol = threadIdx.x;

  const int globalRow = bRow * TILE + tRow;
  const int globalCol = bCol * TILE + tCol;

  float acc = 0.0f;

  int numTiles = CEIL_DIV(K, TILE);
  int cur = 0; // current buffer index

  // Pre-load first tile
  if (globalRow < M && tCol < K)
    As[cur][tRow][tCol] = A[globalRow * K + tCol];
  else
    As[cur][tRow][tCol] = 0.0f;

  if (tRow < K && globalCol < N)
    Bs[cur][tRow][tCol] = B[tRow * N + globalCol];
  else
    Bs[cur][tRow][tCol] = 0.0f;

  __syncthreads();

  for (int t = 0; t < numTiles; ++t) {
    int next = 1 - cur;

    // Pre-load next tile (if exists)
    if (t + 1 < numTiles) {
      int nextTileStart = (t + 1) * TILE;
      if (globalRow < M && nextTileStart + tCol < K)
        As[next][tRow][tCol] = A[globalRow * K + nextTileStart + tCol];
      else
        As[next][tRow][tCol] = 0.0f;

      if (nextTileStart + tRow < K && globalCol < N)
        Bs[next][tRow][tCol] = B[(nextTileStart + tRow) * N + globalCol];
      else
        Bs[next][tRow][tCol] = 0.0f;
    }

    // Compute from current buffer
    for (int k = 0; k < TILE; ++k) {
      acc += As[cur][tRow][k] * Bs[cur][k][tCol];
    }

    __syncthreads();
    cur = next;
  }

  if (globalRow < M && globalCol < N) {
    C[globalRow * N + globalCol] = alpha * acc + beta * C[globalRow * N + globalCol];
  }
}

// Wrapper for transpose-multiply approach
inline void launch_kernel_transpose(int M, int N, int K, float alpha,
                                    const float *A, const float *B,
                                    float beta, float *C) {
  // Step 1: Transpose B (K×N) → Bt (N×K)
  float *Bt;
  CUDA_CHECK(cudaMalloc(&Bt, (size_t)N * K * sizeof(float)));

  dim3 tBlock(32, 32);
  dim3 tGrid(CEIL_DIV(N, 32), CEIL_DIV(K, 32));
  transpose_kernel<<<tGrid, tBlock>>>(B, Bt, K, N);

  // Step 2: Matmul using transposed B
  dim3 mBlock(32, 32);
  dim3 mGrid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_with_transpose<<<mGrid, mBlock>>>(M, N, K, alpha, A, Bt, beta, C);

  CUDA_CHECK(cudaFree(Bt));
}

// Wrapper for recursive tiled approach
inline void launch_kernel_recursive(int M, int N, int K, float alpha,
                                    const float *A, const float *B,
                                    float beta, float *C) {
  const int TILE = 32;
  dim3 block(TILE, TILE);
  dim3 grid(CEIL_DIV(M, TILE), CEIL_DIV(N, TILE));
  sgemm_recursive_tile<TILE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
