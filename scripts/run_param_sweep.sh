#!/bin/bash
# =============================================================================
# run_param_sweep.sh – Parameter sensitivity analysis
#
# Compiles and runs the 2D blocktile kernel with different template parameters
# to identify performance cliffs and optimal configurations.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
SRC_DIR="$PROJECT_DIR/src"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
RESULTS_DIR="$PROJECT_DIR/results/$GPU_NAME"
mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "Parameter Sensitivity Sweep"
echo "GPU: $GPU_NAME (SM $GPU_ARCH)"
echo "============================================="

# Generate a parameter sweep program
SWEEP_SRC="$BUILD_DIR/param_sweep.cu"
SWEEP_BIN="$BUILD_DIR/param_sweep"

cat > "$SWEEP_SRC" << 'SWEEP_EOF'
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../src/kernels/kernel_common.cuh"
#include "../src/kernels/kernel_5_2d_blocktile.cuh"
#include "../src/kernels/kernel_cublas.cuh"

// Instantiate various 2D blocktile configurations
// (BM, BN, BK, TM, TN)

template <int BM, int BN, int BK, int TM, int TN>
float bench_config(int M, int N, int K, const float *dA, const float *dB,
                   float *dC, int runs) {
  int threadsPerBlock = (BM / TM) * (BN / TN);
  if (threadsPerBlock > 1024 || threadsPerBlock <= 0) return -1.0f;

  float alpha = 1.0f, beta = 0.0f;
  // Warmup
  for (int i = 0; i < 3; i++) {
    cudaMemset(dC, 0, (size_t)M * N * sizeof(float));
    dim3 block(threadsPerBlock);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_2d_blocktile<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  cudaDeviceSynchronize();

  GpuTimer timer;
  float total = 0;
  for (int i = 0; i < runs; i++) {
    cudaMemset(dC, 0, (size_t)M * N * sizeof(float));
    timer.tic();
    dim3 block(threadsPerBlock);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_2d_blocktile<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    total += timer.toc();
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return -1.0f;
  }
  return total / runs;
}

int main() {
  srand(42);
  int M = 2048, N = 2048, K = 2048;

  float *hA = new float[M * K];
  float *hB = new float[K * N];
  for (int i = 0; i < M * K; i++) hA[i] = ((float)rand() / RAND_MAX) * 2 - 1;
  for (int i = 0; i < K * N; i++) hB[i] = ((float)rand() / RAND_MAX) * 2 - 1;

  float *dA, *dB, *dC;
  cudaMalloc(&dA, (size_t)M * K * sizeof(float));
  cudaMalloc(&dB, (size_t)K * N * sizeof(float));
  cudaMalloc(&dC, (size_t)M * N * sizeof(float));
  cudaMemcpy(dA, hA, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice);

  int runs = 10;

  char gpu_name[256];
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  snprintf(gpu_name, sizeof(gpu_name), "%s", prop.name);
  for (int i = 0; gpu_name[i]; i++) if (gpu_name[i] == ' ') gpu_name[i] = '_';

  printf("gpu,BM,BN,BK,TM,TN,threads_per_block,M,N,K,time_ms,gflops\n");

  // Macro to test a config
  #define TEST_CONFIG(bm, bn, bk, tm, tn) do { \
    if ((bm) % (tm) == 0 && (bn) % (tn) == 0 && ((bm)/(tm))*((bn)/(tn)) <= 1024) { \
      float ms = bench_config<bm, bn, bk, tm, tn>(M, N, K, dA, dB, dC, runs); \
      if (ms > 0) { \
        double gf = 2.0 * (double)M * N * K / (ms * 1e6); \
        printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.2f\n", \
               gpu_name, bm, bn, bk, tm, tn, ((bm)/(tm))*((bn)/(tn)), M, N, K, ms, gf); \
        fprintf(stderr, "  BM=%d BN=%d BK=%d TM=%d TN=%d -> %.2f GFLOPS\n", \
                bm, bn, bk, tm, tn, gf); \
      } \
    } \
  } while(0)

  // Sweep BK with fixed BM=BN=128, TM=TN=8
  fprintf(stderr, "=== Sweeping BK ===\n");
  TEST_CONFIG(128, 128, 4, 8, 8);
  TEST_CONFIG(128, 128, 8, 8, 8);
  TEST_CONFIG(128, 128, 16, 8, 8);
  TEST_CONFIG(128, 128, 32, 8, 8);

  // Sweep TM, TN with fixed BM=BN=128, BK=8
  fprintf(stderr, "=== Sweeping TM/TN ===\n");
  TEST_CONFIG(128, 128, 8, 4, 4);
  TEST_CONFIG(128, 128, 8, 4, 8);
  TEST_CONFIG(128, 128, 8, 8, 4);
  TEST_CONFIG(128, 128, 8, 8, 8);
  TEST_CONFIG(128, 128, 8, 16, 8);
  TEST_CONFIG(128, 128, 8, 8, 16);
  TEST_CONFIG(128, 128, 8, 16, 16);

  // Sweep BM, BN with fixed BK=8, TM=TN=8
  fprintf(stderr, "=== Sweeping BM/BN ===\n");
  TEST_CONFIG(32, 32, 8, 8, 8);   // Only 16 threads - may be too few
  TEST_CONFIG(64, 64, 8, 8, 8);
  TEST_CONFIG(64, 128, 8, 8, 8);
  TEST_CONFIG(128, 64, 8, 8, 8);
  TEST_CONFIG(128, 128, 8, 8, 8);
  TEST_CONFIG(128, 256, 8, 8, 8);
  TEST_CONFIG(256, 128, 8, 8, 8);

  // Extreme configs to show performance cliffs
  fprintf(stderr, "=== Edge cases ===\n");
  TEST_CONFIG(64, 64, 4, 4, 4);
  TEST_CONFIG(64, 64, 16, 4, 4);
  TEST_CONFIG(256, 256, 8, 8, 8);  // 1024 threads
  TEST_CONFIG(128, 128, 8, 4, 4);  // 1024 threads

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  delete[] hA;
  delete[] hB;
  return 0;
}
SWEEP_EOF

echo ">>> Compiling parameter sweep program..."
nvcc -O3 -std=c++17 -arch=sm_"$GPU_ARCH" --use_fast_math -lineinfo \
    -o "$SWEEP_BIN" "$SWEEP_SRC" -lcublas \
    -I"$PROJECT_DIR"

echo ">>> Running parameter sweep..."
"$SWEEP_BIN" 2>"$RESULTS_DIR/param_sweep.log" | tee "$RESULTS_DIR/param_sweep.csv"

echo ""
echo "============================================="
echo "Parameter sweep complete!"
echo "Results: $RESULTS_DIR/param_sweep.csv"
echo "============================================="
