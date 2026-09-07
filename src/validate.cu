/*
 * validate.cu – Correctness validation for all SGEMM kernels
 *
 * Runs each kernel on several matrix sizes and compares output to cuBLAS.
 * Exits with non-zero status if any kernel exceeds error tolerance.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "kernels/kernel_common.cuh"
#include "kernels/kernel_1_naive.cuh"
#include "kernels/kernel_2_coalescing.cuh"
#include "kernels/kernel_3_smem.cuh"
#include "kernels/kernel_4_1d_blocktile.cuh"
#include "kernels/kernel_5_2d_blocktile.cuh"
#include "kernels/kernel_6_vectorized.cuh"
#include "kernels/kernel_7_bank_conflicts.cuh"
#include "kernels/kernel_8_warptiling.cuh"
#include "kernels/kernel_9_double_buffering.cuh"
#include "kernels/kernel_cublas.cuh"
#include "kernels/kernel_extra.cuh"

typedef void (*LaunchFn)(int, int, int, float, const float *, const float *,
                         float, float *);

struct KernelEntry {
  const char *name;
  LaunchFn launch;
};

// Dispatch wrapper
void dispatch(KernelId kid, int M, int N, int K, float alpha, const float *A,
              const float *B, float beta, float *C) {
  switch (kid) {
  case KERNEL_NAIVE: launch_kernel_naive(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_COALESCING: launch_kernel_coalescing(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_SMEM: launch_kernel_smem(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_1D_BLOCKTILE: launch_kernel_1d_blocktile(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_2D_BLOCKTILE: launch_kernel_2d_blocktile(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_VECTORIZED: launch_kernel_vectorized(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_RESOLVE_BANK_CONFLICTS: launch_kernel_bank_conflicts(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_WARPTILING: launch_kernel_warptiling(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_DOUBLE_BUFFERING: launch_kernel_double_buffering(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_EXTRA_TRANSPOSE: launch_kernel_transpose(M, N, K, alpha, A, B, beta, C); break;
  case KERNEL_EXTRA_RECURSIVE: launch_kernel_recursive(M, N, K, alpha, A, B, beta, C); break;
  default: break;
  }
}

int main() {
  srand(42);
  cublas_init();

  struct TestSize { int M, N, K; };
  TestSize sizes[] = {
    {64, 64, 64},
    {128, 128, 128},
    {256, 256, 256},
    {512, 512, 512},
    {1024, 1024, 1024},
    // Non-square
    {128, 256, 512},
    {300, 150, 300},
    {1000, 500, 750},
    // Non-power-of-2
    {127, 127, 127},
    {255, 255, 255},
    {513, 513, 513},
  };
  int numSizes = sizeof(sizes) / sizeof(sizes[0]);

  KernelId kernels[] = {
    KERNEL_NAIVE, KERNEL_COALESCING, KERNEL_SMEM,
    KERNEL_1D_BLOCKTILE, KERNEL_2D_BLOCKTILE, KERNEL_VECTORIZED,
    KERNEL_RESOLVE_BANK_CONFLICTS, KERNEL_WARPTILING, KERNEL_DOUBLE_BUFFERING,
    KERNEL_EXTRA_TRANSPOSE, KERNEL_EXTRA_RECURSIVE
  };
  int numKernels = sizeof(kernels) / sizeof(kernels[0]);

  int total_tests = 0, passed = 0, failed = 0;
  float tolerance = 1e-2f; // Relative tolerance for large matrices

  for (int si = 0; si < numSizes; ++si) {
    int M = sizes[si].M, N = sizes[si].N, K = sizes[si].K;
    size_t sA = (size_t)M * K * sizeof(float);
    size_t sB = (size_t)K * N * sizeof(float);
    size_t sC = (size_t)M * N * sizeof(float);

    // Host matrices
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC_ref = new float[M * N];
    float *hC = new float[M * N];
    randomize_matrix(hA, M, K);
    randomize_matrix(hB, K, N);

    // Device matrices
    float *dA, *dB, *dC, *dC_ref;
    CUDA_CHECK(cudaMalloc(&dA, sA));
    CUDA_CHECK(cudaMalloc(&dB, sB));
    CUDA_CHECK(cudaMalloc(&dC, sC));
    CUDA_CHECK(cudaMalloc(&dC_ref, sC));
    CUDA_CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));

    // Compute reference with cuBLAS
    CUDA_CHECK(cudaMemset(dC_ref, 0, sC));
    launch_kernel_cublas(M, N, K, 1.0f, dA, dB, 0.0f, dC_ref);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC_ref, dC_ref, sC, cudaMemcpyDeviceToHost));

    for (int ki = 0; ki < numKernels; ++ki) {
      total_tests++;
      CUDA_CHECK(cudaMemset(dC, 0, sC));
      dispatch(kernels[ki], M, N, K, 1.0f, dA, dB, 0.0f, dC);
      CUDA_CHECK(cudaDeviceSynchronize());

      // Check for CUDA errors from kernel launch
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("[FAIL] %-25s  %4dx%4dx%4d  CUDA error: %s\n",
               kernel_name(kernels[ki]), M, N, K, cudaGetErrorString(err));
        failed++;
        continue;
      }

      CUDA_CHECK(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));

      float merr = max_error(hC, hC_ref, M, N);

      // Scale tolerance with matrix size (FP32 accumulation error grows)
      float tol = tolerance * sqrtf((float)K);

      if (merr < tol) {
        printf("[PASS] %-25s  %4dx%4dx%4d  max_err=%.2e\n",
               kernel_name(kernels[ki]), M, N, K, merr);
        passed++;
      } else {
        printf("[FAIL] %-25s  %4dx%4dx%4d  max_err=%.2e (tol=%.2e)\n",
               kernel_name(kernels[ki]), M, N, K, merr, tol);
        failed++;
      }
    }

    // Cleanup per size
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dC_ref));
    delete[] hA;
    delete[] hB;
    delete[] hC_ref;
    delete[] hC;
  }

  cublas_destroy();

  printf("\n========================================\n");
  printf("Total: %d  Passed: %d  Failed: %d\n", total_tests, passed, failed);
  printf("========================================\n");

  return (failed > 0) ? 1 : 0;
}
