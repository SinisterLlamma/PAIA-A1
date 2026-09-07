#!/usr/bin/env python3
"""
parse_ncu_reports.py
Parses all .ncu-rep files in results/<GPU>/ncu_profiles/ into a clean ncu_metrics.csv.
"""
import os
import glob
import subprocess
import csv

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")

    # Detect GPU directory
    gpu_dirs = [d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d) and not os.path.basename(d).startswith('.')]
    if not gpu_dirs:
        print("No GPU results directory found.")
        return

    gpu_dir = gpu_dirs[0]
    gpu_name = os.path.basename(gpu_dir)
    ncu_dir = os.path.join(gpu_dir, "ncu_profiles")
    out_csv = os.path.join(gpu_dir, "ncu_metrics.csv")

    ncu_bin = "/opt/nvidia/nsight-compute/2024.3.2/ncu"
    if not os.path.exists(ncu_bin):
        ncu_bin = "ncu"

    rep_files = sorted(glob.glob(os.path.join(ncu_dir, "*.ncu-rep")))
    if not rep_files:
        print("No .ncu-rep files found in", ncu_dir)
        return

    kernel_names = [
        "cuBLAS", "1_Naive", "2_GMEM_Coalescing", "3_SMEM_Caching",
        "4_1D_Blocktile", "5_2D_Blocktile", "6_Vectorized", "7_Bank_Extra_Col",
        "8_Warptiling", "9_Double_Buffering", "10_Transpose", "11_Recursive_Tile"
    ]

    rows = []
    header = [
        "gpu", "kernel_id", "kernel_name", "M", "N", "K",
        "sm_throughput_pct", "dram_throughput_pct",
        "l2_hit_rate_pct", "l1_hit_rate_pct",
        "bank_conflicts_ld", "bank_conflicts_st",
        "registers_per_thread", "occupancy_pct"
    ]

    for kid, kname in enumerate(kernel_names):
        rep_file = os.path.join(ncu_dir, f"kernel_{kid}_{kname}.ncu-rep")
        if not os.path.exists(rep_file):
            # Try alternate pattern
            matches = glob.glob(os.path.join(ncu_dir, f"kernel_{kid}_*.ncu-rep"))
            if matches:
                rep_file = matches[0]
            else:
                continue

        csv_file = os.path.join(ncu_dir, f"kernel_{kid}_{kname}.csv")
        # Export ncu-rep to csv
        cmd = [ncu_bin, "--import", rep_file, "--csv"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not res.stdout.strip():
            print(f"Warning: Failed to export {rep_file}")
            continue

        with open(csv_file, "w") as f:
            f.write(res.stdout)

        # Parse CSV lines
        reader = csv.DictReader(res.stdout.splitlines())
        metrics_by_kernel = {}
        for r in reader:
            k_id = r.get("ID")
            metric = r.get("Metric Name")
            val = r.get("Metric Value", "").replace(",", "")
            if k_id not in metrics_by_kernel:
                metrics_by_kernel[k_id] = {"name": r.get("Kernel Name", "")}
            metrics_by_kernel[k_id][metric] = val

        # Select target kernel: if kid == 0, select kernel 0. Otherwise select last kernel
        target_id = "0" if kid == 0 else str(max(int(k) for k in metrics_by_kernel.keys() if k.isdigit()))
        m = metrics_by_kernel.get(target_id, {})

        sm_tp = m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "N/A")
        dram_tp = m.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", "N/A")
        l2_hr = m.get("lts__t_sector_hit_rate.pct", "N/A")
        l1_hr = m.get("l1tex__t_sector_hit_rate.pct", "N/A")
        bc_ld = m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "0")
        bc_st = m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "0")
        regs = m.get("launch__registers_per_thread", "N/A")
        occ = m.get("sm__warps_active.avg.pct_of_peak_sustained_active", "N/A")

        rows.append([
            gpu_name, kid, kname, 512, 512, 512,
            sm_tp, dram_tp, l2_hr, l1_hr, bc_ld, bc_st, regs, occ
        ])
        print(f"Parsed Kernel {kid} ({kname}): Regs={regs}, Occ={occ}%, SM={sm_tp}%, DRAM={dram_tp}%, L2Hit={l2_hr}%, BankConfLd={bc_ld}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} kernel records to {out_csv}")

if __name__ == "__main__":
    main()
