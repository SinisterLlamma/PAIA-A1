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

# Check if ncu is available
if ! command -v ncu &>/dev/null; then
    echo "ERROR: ncu (Nsight Compute) not found in PATH"
    echo "Install it from: https://developer.nvidia.com/nsight-compute"
    exit 1
fi

# Metrics to collect
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS+=",dram__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS+=",l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second"
METRICS+=",l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second"
METRICS+=",lts__t_sectors_op_read.sum"
METRICS+=",lts__t_sectors_op_write.sum"
METRICS+=",lts__t_sector_hit_rate.pct"
METRICS+=",l1tex__t_sector_hit_rate.pct"
METRICS+=",l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
METRICS+=",l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
METRICS+=",launch__registers_per_thread"
METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active"
METRICS+=",sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
METRICS+=",sm__sass_thread_inst_executed_op_fmul_pred_on.sum"
METRICS+=",sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
METRICS+=",dram__bytes_read.sum"
METRICS+=",dram__bytes_write.sum"

# Size for profiling (smaller to keep ncu runs manageable)
PROF_M=1024
PROF_N=1024
PROF_K=1024

# CSV header for summary
NCU_SUMMARY="$RESULTS_DIR/ncu_metrics.csv"
echo "gpu,kernel_id,kernel_name,M,N,K,sm_throughput_pct,dram_throughput_pct,l2_hit_rate_pct,l1_hit_rate_pct,bank_conflicts_ld,bank_conflicts_st,registers_per_thread,occupancy_pct,dram_bytes_read,dram_bytes_write" > "$NCU_SUMMARY"

KERNEL_NAMES=("cuBLAS" "1_Naive" "2_GMEM_Coalescing" "3_SMEM_Caching" "4_1D_Blocktile" "5_2D_Blocktile" "6_Vectorized" "7_Bank_Extra_Col" "8_Warptiling" "9_Double_Buffering" "10_Transpose" "11_Recursive_Tile")

for kid in 0 1 2 3 4 5 6 7 8 9 10 11; do
    kname="${KERNEL_NAMES[$kid]}"
    echo ""
    echo ">>> Profiling kernel $kid: $kname"

    NCU_REPORT="$NCU_DIR/kernel_${kid}_${kname}.ncu-rep"
    NCU_CSV="$NCU_DIR/kernel_${kid}_${kname}.csv"

    # Run ncu - profile only 1 run, skip warmup kernels
    # --kernel-id selects which kernel launch to profile
    # We profile the last kernel launch (the actual benchmark, not warmup)
    ncu --set full \
        --metrics "$METRICS" \
        --csv \
        --target-processes all \
        --export "$NCU_REPORT" \
        "$BENCHMARK" -k "$kid" -m "$PROF_M" -n "$PROF_N" -k_dim "$PROF_K" --warmup 0 --runs 1 \
        2>/dev/null \
        > "$NCU_CSV" || {
            echo "  WARNING: ncu failed for kernel $kid. Try running with sudo."
            echo "  sudo ncu ... or: sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0"
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
        DRAM_R=$(grep "dram__bytes_read" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "N/A")
        DRAM_W=$(grep "dram__bytes_write" "$NCU_CSV" | tail -1 | awk -F',' '{print $NF}' | tr -d '"' || echo "N/A")

        echo "$GPU_NAME,$kid,$kname,$PROF_M,$PROF_N,$PROF_K,$SM_TP,$DRAM_TP,$L2_HR,$L1_HR,$BC_LD,$BC_ST,$REGS,$OCC,$DRAM_R,$DRAM_W" >> "$NCU_SUMMARY"
        echo "  ✓ Saved: $NCU_CSV"
    fi
done

echo ""
echo "============================================="
echo "Nsight Compute profiling complete!"
echo "Summary: $NCU_SUMMARY"
echo "Reports: $NCU_DIR/"
echo "============================================="
