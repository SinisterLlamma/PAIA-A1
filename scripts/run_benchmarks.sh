#!/bin/bash
# =============================================================================
# run_benchmarks.sh – Full benchmark suite
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
BENCHMARK="$BUILD_DIR/benchmark"

# Auto-detect GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
RESULTS_DIR="$PROJECT_DIR/results/$GPU_NAME"
mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "CUDA SGEMM Benchmark Suite"
echo "GPU: $GPU_NAME"
echo "Results: $RESULTS_DIR/"
echo "============================================="

# Build if needed
if [ ! -f "$BENCHMARK" ]; then
    echo "Building benchmark..."
    make -C "$PROJECT_DIR" benchmark
fi

# 1. Full dimension sweep
echo ""
echo ">>> Running full dimension sweep..."
"$BENCHMARK" --sweep --runs 20 \
    2>"$RESULTS_DIR/sweep.log" \
    | tee "$RESULTS_DIR/benchmarks.csv"

echo ""
echo ">>> Benchmark results saved to: $RESULTS_DIR/benchmarks.csv"

# 2. Individual kernel runs for specific sizes (for profiling targets)
echo ""
echo ">>> Running per-kernel benchmarks for profiling sizes..."
for kid in 0 1 2 3 4 5 6 7 8 9 10 11; do
    "$BENCHMARK" -k "$kid" -m 2048 -n 2048 -k_dim 2048 --runs 5 \
        2>/dev/null | tail -1
done | tee -a "$RESULTS_DIR/benchmarks_2048.csv"

echo ""
echo "============================================="
echo "Benchmarks complete!"
echo "============================================="
