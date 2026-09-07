#!/bin/bash
# =============================================================================
# collect_gpu_info.sh – Collect GPU hardware specifications
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
RESULTS_DIR="$PROJECT_DIR/results/$GPU_NAME"
mkdir -p "$RESULTS_DIR"

INFO_FILE="$RESULTS_DIR/gpu_info.txt"

echo "=============================================" > "$INFO_FILE"
echo "GPU Hardware Information" >> "$INFO_FILE"
echo "Collected: $(date)" >> "$INFO_FILE"
echo "=============================================" >> "$INFO_FILE"

echo "" >> "$INFO_FILE"
echo "=== nvidia-smi ===" >> "$INFO_FILE"
nvidia-smi >> "$INFO_FILE" 2>&1

echo "" >> "$INFO_FILE"
echo "=== nvidia-smi -q (detailed) ===" >> "$INFO_FILE"
nvidia-smi -q >> "$INFO_FILE" 2>&1

echo "" >> "$INFO_FILE"
echo "=== CUDA Device Properties ===" >> "$INFO_FILE"
# Use a simple CUDA program to query device properties
QUERY_SRC="$PROJECT_DIR/build/query_device.cu"
QUERY_BIN="$PROJECT_DIR/build/query_device"
mkdir -p "$PROJECT_DIR/build"

cat > "$QUERY_SRC" << 'EOF'
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability:           %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory:          %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Shared Memory per Block:      %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Shared Memory per SM:         %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  Registers per Block:          %d\n", prop.regsPerBlock);
        printf("  Registers per SM:             %d\n", prop.regsPerMultiprocessor);
        printf("  Warp Size:                    %d\n", prop.warpSize);
        printf("  Max Threads per Block:        %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per SM:           %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Warps per SM:             %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
        printf("  Number of SMs:                %d\n", prop.multiProcessorCount);
        printf("  Memory Bus Width:             %d bits\n", prop.memoryBusWidth);
        printf("  Memory Clock Rate:            %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Peak Memory Bandwidth:        %.2f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  L2 Cache Size:                %.2f MB\n", prop.l2CacheSize / 1e6);
        printf("  Clock Rate:                   %.2f GHz\n", prop.clockRate / 1e6);
        printf("  Peak FP32 (est):              %.2f TFLOPS\n",
               2.0 * prop.multiProcessorCount * (prop.clockRate / 1e6) * 128 / 1e3);
        printf("  Concurrent Kernels:           %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled:                  %s\n", prop.ECCEnabled ? "Yes" : "No");
    }
    return 0;
}
EOF

GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
nvcc -o "$QUERY_BIN" "$QUERY_SRC" -arch=sm_"$GPU_ARCH" 2>/dev/null
"$QUERY_BIN" >> "$INFO_FILE" 2>&1

echo "" >> "$INFO_FILE"
echo "=== NVCC Version ===" >> "$INFO_FILE"
nvcc --version >> "$INFO_FILE" 2>&1

echo "" >> "$INFO_FILE"
echo "=== CUDA Driver Version ===" >> "$INFO_FILE"
nvidia-smi --query-gpu=driver_version --format=csv,noheader >> "$INFO_FILE" 2>&1

echo ""
echo "GPU info saved to: $INFO_FILE"
cat "$INFO_FILE"
