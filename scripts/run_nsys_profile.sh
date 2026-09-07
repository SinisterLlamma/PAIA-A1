#!/bin/bash
# =============================================================================
# run_nsys_profile.sh – Nsight Systems timeline profiling
#
# Generates timeline traces (.nsys-rep) showing kernel execution,
# memory transfers, and CPU-GPU interaction.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
BENCHMARK="$BUILD_DIR/benchmark"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
RESULTS_DIR="$PROJECT_DIR/results/$GPU_NAME"
NSYS_DIR="$RESULTS_DIR/nsys_traces"
mkdir -p "$NSYS_DIR"

echo "============================================="
echo "Nsight Systems Profiling"
echo "GPU: $GPU_NAME"
echo "============================================="

if [ ! -f "$BENCHMARK" ]; then
    make -C "$PROJECT_DIR" benchmark
fi

if ! command -v nsys &>/dev/null; then
    echo "ERROR: nsys (Nsight Systems) not found in PATH"
    exit 1
fi

# Profile the full sweep
echo ">>> Profiling full benchmark sweep..."
nsys profile \
    --output "$NSYS_DIR/full_sweep" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --stats true \
    "$BENCHMARK" -k all -s 1024,2048 --runs 3 \
    2>&1 | tee "$NSYS_DIR/nsys_full.log"

# Profile individual kernels at 2048x2048
for kid in 1 2 3 4 5 6 7 8 9; do
    echo ""
    echo ">>> Profiling kernel $kid at 2048x2048..."
    nsys profile \
        --output "$NSYS_DIR/kernel_${kid}" \
        --force-overwrite true \
        --trace cuda \
        --stats true \
        "$BENCHMARK" -k "$kid" -m 2048 -n 2048 -k_dim 2048 --runs 3 \
        2>&1 | tee "$NSYS_DIR/nsys_kernel_${kid}.log"
done

# Generate summary statistics
echo ""
echo ">>> Generating summary statistics..."
nsys stats "$NSYS_DIR/full_sweep.nsys-rep" \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$NSYS_DIR/kernel_summary" \
    2>/dev/null || echo "  Note: nsys stats may require GUI version"

echo ""
echo "============================================="
echo "Nsight Systems profiling complete!"
echo "Traces: $NSYS_DIR/"
echo ""
echo "To view traces:"
echo "  nsys-ui $NSYS_DIR/full_sweep.nsys-rep"
echo "============================================="
