#!/bin/bash
# =============================================================================
# run_ncu_profile.sh – Nsight Compute profiling for each kernel
#
# Collects detailed hardware metrics: occupancy, memory throughput,
# L1/L2 cache hit rates, shared memory bank conflicts, register usage.
#
# NOTE: ncu often requires root or CAP_SYS_ADMIN. If it fails, try:
#   sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
# or run with sudo.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
BENCHMARK="$BUILD_DIR/benchmark"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
RESULTS_DIR="$PROJECT_DIR/results/$GPU_NAME"
NCU_DIR="$RESULTS_DIR/ncu_profiles"
mkdir -p "$NCU_DIR"

echo "============================================="
echo "Nsight Compute Profiling"
echo "GPU: $GPU_NAME"
echo "============================================="

# Build if needed
if [ ! -f "$BENCHMARK" ]; then
    make -C "$PROJECT_DIR" benchmark
fi

# Find ncu binary
NCU_BIN=""
if command -v ncu &>/dev/null; then
    NCU_BIN="ncu"
elif [ -x "/opt/nvidia/nsight-compute/2024.3.2/ncu" ]; then
    NCU_BIN="/opt/nvidia/nsight-compute/2024.3.2/ncu"
elif [ -x "/usr/local/cuda-12.6/bin/ncu" ]; then
    NCU_BIN="/usr/local/cuda-12.6/bin/ncu"
else
    echo "ERROR: ncu (Nsight Compute) not found."
    exit 1
fi
echo "Using NCU binary: $NCU_BIN"

# Metrics to collect
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS+=",dram__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS+=",lts__t_sector_hit_rate.pct"
METRICS+=",l1tex__t_sector_hit_rate.pct"
METRICS+=",l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
METRICS+=",l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
METRICS+=",launch__registers_per_thread"
METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active"

# Size for profiling (512x512 keeps ncu multi-pass runs very fast)
PROF_M=512
PROF_N=512
PROF_K=512

# CSV header for summary
NCU_SUMMARY="$RESULTS_DIR/ncu_metrics.csv"
echo "gpu,kernel_id,kernel_name,M,N,K,sm_throughput_pct,dram_throughput_pct,l2_hit_rate_pct,l1_hit_rate_pct,bank_conflicts_ld,bank_conflicts_st,registers_per_thread,occupancy_pct" > "$NCU_SUMMARY"

KERNEL_NAMES=("cuBLAS" "1_Naive" "2_GMEM_Coalescing" "3_SMEM_Caching" "4_1D_Blocktile" "5_2D_Blocktile" "6_Vectorized" "7_Bank_Extra_Col" "8_Warptiling" "9_Double_Buffering" "10_Transpose" "11_Recursive_Tile")

for kid in 0 1 2 3 4 5 6 7 8 9 10 11; do
    kname="${KERNEL_NAMES[$kid]}"
    echo ""
    echo ">>> Profiling kernel $kid: $kname"

    NCU_REPORT="$NCU_DIR/kernel_${kid}_${kname}"
    NCU_CSV="$NCU_DIR/kernel_${kid}_${kname}.csv"

    "$NCU_BIN" \
        --metrics "$METRICS" \
        --csv \
        --target-processes all \
        -f -o "$NCU_REPORT" \
        "$BENCHMARK" -k "$kid" -m "$PROF_M" -n "$PROF_N" -k_dim "$PROF_K" --warmup 0 --runs 1 \
        > "$NCU_CSV" 2>&1 || {
            echo "  WARNING: ncu failed for kernel $kid."
            continue
        }

    # Parse the CSV to extract key metrics
    if [ -f "$NCU_CSV" ] && [ -s "$NCU_CSV" ]; then
        # Extract metrics from the CSV (last kernel invocation)
        SM_TP=$(grep "sm__throughput" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "N/A")
        DRAM_TP=$(grep "dram__throughput" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "N/A")
        L2_HR=$(grep "lts__t_sector_hit_rate" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "N/A")
        L1_HR=$(grep "l1tex__t_sector_hit_rate" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "N/A")
        BC_LD=$(grep "bank_conflicts.*op_ld" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "0")
        BC_ST=$(grep "bank_conflicts.*op_st" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "0")
        REGS=$(grep "registers_per_thread" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "N/A")
        OCC=$(grep "warps_active.*pct" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "N/A")
        echo "$GPU_NAME,$kid,$kname,$PROF_M,$PROF_N,$PROF_K,$SM_TP,$DRAM_TP,$L2_HR,$L1_HR,$BC_LD,$BC_ST,$REGS,$OCC" >> "$NCU_SUMMARY"
        echo "  ✓ Saved: $NCU_CSV"
    fi
done

echo ""
echo "============================================="
echo "Nsight Compute profiling complete!"
echo "Summary: $NCU_SUMMARY"
echo "Reports: $NCU_DIR/"
echo "============================================="
