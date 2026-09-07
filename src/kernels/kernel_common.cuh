#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Utility macros
// ---------------------------------------------------------------------------
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// CUDA error checking – wraps every runtime call
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// GPU Timer using CUDA events (microsecond-accurate)
// ---------------------------------------------------------------------------
struct GpuTimer {
  cudaEvent_t start, stop;

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void tic() { CUDA_CHECK(cudaEventRecord(start, 0)); }

  // Returns elapsed time in milliseconds
  float toc() {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

// Fill matrix with random values in [-1, 1]
inline void randomize_matrix(float *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
}

// Fill matrix with zeros
inline void zero_matrix(float *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = 0.0f;
  }
}

// Copy matrix
inline void copy_matrix(const float *src, float *dst, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    dst[i] = src[i];
  }
}

// Check two matrices for approximate equality, returns max absolute error
inline float max_error(const float *A, const float *B, int rows, int cols) {
  float max_err = 0.0f;
  for (int i = 0; i < rows * cols; i++) {
    float err = fabsf(A[i] - B[i]);
    if (err > max_err) max_err = err;
  }
  return max_err;
}

// Compute GFLOPS for GEMM: C = alpha*A*B + beta*C
// FLOPs = 2*M*N*K (multiply + add per element of dot product)
inline double compute_gflops(int M, int N, int K, float time_ms) {
  double flops = 2.0 * (double)M * (double)N * (double)K;
  return flops / (time_ms * 1e6); // ms -> s, then / 1e9
}

// ---------------------------------------------------------------------------
// Kernel ID enum for dispatch (Simon Boehm progression)
// ---------------------------------------------------------------------------
enum KernelId {
  KERNEL_CUBLAS = 0,
  KERNEL_NAIVE = 1,
  KERNEL_COALESCING = 2,
  KERNEL_SMEM = 3,
  KERNEL_1D_BLOCKTILE = 4,
  KERNEL_2D_BLOCKTILE = 5,
  KERNEL_VECTORIZED = 6,
  KERNEL_RESOLVE_BANK_CONFLICTS = 7,
  KERNEL_WARPTILING = 8,
  KERNEL_DOUBLE_BUFFERING = 9,
  KERNEL_EXTRA_TRANSPOSE = 10,
  KERNEL_EXTRA_RECURSIVE = 11,
  KERNEL_COUNT = 12
};

inline const char *kernel_name(KernelId id) {
  switch (id) {
  case KERNEL_CUBLAS:
    return "cuBLAS";
  case KERNEL_NAIVE:
    return "1_Naive";
  case KERNEL_COALESCING:
    return "2_GMEM_Coalescing";
  case KERNEL_SMEM:
    return "3_SMEM_Caching";
  case KERNEL_1D_BLOCKTILE:
    return "4_1D_Blocktile";
  case KERNEL_2D_BLOCKTILE:
    return "5_2D_Blocktile";
  case KERNEL_VECTORIZED:
    return "6_Vectorized";
  case KERNEL_RESOLVE_BANK_CONFLICTS:
    return "7_Bank_Extra_Col";
  case KERNEL_WARPTILING:
    return "8_Warptiling";
  case KERNEL_DOUBLE_BUFFERING:
    return "9_Double_Buffering";
  case KERNEL_EXTRA_TRANSPOSE:
    return "10_Transpose";
  case KERNEL_EXTRA_RECURSIVE:
    return "11_Recursive_Tile";
  default:
    return "Unknown";
  }
}
