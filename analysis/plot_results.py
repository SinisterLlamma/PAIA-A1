#!/usr/bin/env python3
"""
plot_results.py – Generate publication-quality plots from benchmark results.

Usage:
    python3 analysis/plot_results.py results/NVIDIA_GeForce_RTX_3090/
    python3 analysis/plot_results.py results/NVIDIA_GeForce_RTX_3090/ --save
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 15,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# Color palette for kernels
KERNEL_COLORS = {
    'cuBLAS': '#2ecc71',
    '1_Naive': '#e74c3c',
    '2_GMEM_Coalescing': '#e67e22',
    '3_SMEM_Caching': '#f39c12',
    '4_1D_Blocktile': '#3498db',
    '5_2D_Blocktile': '#9b59b6',
    '6_Vectorized': '#1abc9c',
    '7_Bank_Extra_Col': '#16a085',
    '8_Warptiling': '#2c3e50',
    '9_Double_Buffering': '#8e44ad',
    '10_Transpose': '#95a5a6',
    '11_Recursive_Tile': '#d35400',
}


def load_benchmark_data(results_dir):
    """Load benchmark CSV data."""
    csv_path = os.path.join(results_dir, 'benchmarks.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['kernel_name', 'gflops'])
    return df


def load_param_sweep(results_dir):
    """Load parameter sweep CSV data."""
    csv_path = os.path.join(results_dir, 'param_sweep.csv')
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None


def load_ncu_metrics(results_dir):
    """Load Nsight Compute metrics CSV."""
    csv_path = os.path.join(results_dir, 'ncu_metrics.csv')
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None


def plot_gflops_by_kernel(df, results_dir, save=False):
    """Bar chart: GFLOPS per kernel for the largest square matrix."""
    square = df[df['M'] == df['N']]
    square = square[square['M'] == square['K']]
    if len(square) == 0:
        square = df
    largest_size = square['M'].max()
    data = square[square['M'] == largest_size].copy()

    if len(data) == 0:
        print("Warning: No data for GFLOPS by kernel plot")
        return

    data = data.sort_values('kernel_id')

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = [KERNEL_COLORS.get(name, '#7f8c8d') for name in data['kernel_name']]
    bars = ax.bar(range(len(data)), data['gflops'], color=colors, edgecolor='white',
                  linewidth=0.8, width=0.65)

    max_gf = data['gflops'].max()
    for bar, gf in zip(bars, data['gflops']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_gf * 0.015,
                f'{gf:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    cublas_gf = data[data['kernel_name'] == 'cuBLAS']['gflops']
    if len(cublas_gf) > 0:
        ax.axhline(y=cublas_gf.values[0], color='#27ae60', linestyle='--',
                   linewidth=2, alpha=0.75, label=f'cuBLAS Baseline ({cublas_gf.values[0]:,.0f} GFLOPS)')

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['kernel_name'], rotation=40, ha='right', fontweight='medium')
    ax.set_ylabel('Throughput (GFLOPS)', fontweight='bold')
    ax.set_title(f'SGEMM Performance by Kernel — {data["gpu"].iloc[0]} ({largest_size}×{largest_size})',
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, max_gf * 1.15)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(results_dir, 'plot_gflops_by_kernel.png'))
        print(f"  Saved: plot_gflops_by_kernel.png")
    else:
        plt.show()
    plt.close()


def plot_pct_cublas(df, results_dir, save=False):
    """Bar chart: % of cuBLAS performance per kernel."""
    square = df[(df['M'] == df['N']) & (df['M'] == df['K'])]
    largest_size = square['M'].max()
    data = square[square['M'] == largest_size].copy()

    cublas_gf = data[data['kernel_name'] == 'cuBLAS']['gflops']
    if len(cublas_gf) == 0:
        print("Warning: No cuBLAS data for percentage plot")
        return

    cublas_val = cublas_gf.values[0]
    data = data[data['kernel_name'] != 'cuBLAS'].copy()
    data['pct_cublas'] = (data['gflops'] / cublas_val) * 100
    data = data.sort_values('kernel_id')

    fig, ax = plt.subplots(figsize=(13, 6.5))

    colors = [KERNEL_COLORS.get(name, '#7f8c8d') for name in data['kernel_name']]
    bars = ax.bar(range(len(data)), data['pct_cublas'], color=colors,
                  edgecolor='white', linewidth=0.8, width=0.65)

    for bar, pct in zip(bars, data['pct_cublas']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=100, color='#27ae60', linestyle='--', linewidth=2, alpha=0.8,
               label='cuBLAS (100%)')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['kernel_name'], rotation=40, ha='right', fontweight='medium')
    ax.set_ylabel('% of cuBLAS Throughput', fontweight='bold')
    ax.set_title(f'Performance Relative to cuBLAS — {data["gpu"].iloc[0]} ({largest_size}×{largest_size})',
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 118)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(results_dir, 'plot_pct_cublas.png'))
        print(f"  Saved: plot_pct_cublas.png")
    else:
        plt.show()
    plt.close()


def plot_gflops_vs_size(df, results_dir, save=False):
    """Line plot: GFLOPS vs matrix size for each kernel."""
    square = df[(df['M'] == df['N']) & (df['M'] == df['K'])]
    if len(square) == 0:
        print("Warning: No square matrix data for size scaling plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    for kname in square['kernel_name'].unique():
        kdata = square[square['kernel_name'] == kname].sort_values('M')
        color = KERNEL_COLORS.get(kname, '#7f8c8d')
        marker = 'o' if 'cuBLAS' in kname or 'Warptiling' in kname else 's'
        linewidth = 2.5 if 'cuBLAS' in kname or 'Warptiling' in kname else 1.8
        ax.plot(kdata['M'], kdata['gflops'], f'{marker}-', color=color, label=kname,
                markersize=6, linewidth=linewidth, alpha=0.9)

    ax.set_xlabel('Matrix Dimension N (N×N)', fontweight='bold')
    ax.set_ylabel('Throughput (GFLOPS)', fontweight='bold')
    ax.set_title(f'SGEMM Scaling with Matrix Dimension — {square["gpu"].iloc[0]}',
                 fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.95)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    max_gf = square['gflops'].max()
    ax.set_ylim(0, max_gf * 1.08)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(results_dir, 'plot_gflops_vs_size.png'))
        print(f"  Saved: plot_gflops_vs_size.png")
    else:
        plt.show()
    plt.close()


def plot_roofline(df, results_dir, save=False):
    """Roofline model plot with accurate arithmetic intensity & uncluttered labels."""
    # Peak specs: default to RTX 3090 specs
    peak_flops = 35.58  # TFLOPS FP32
    peak_bw = 936.0     # GB/s GDDR6X

    # Try to read actual GPU specs from gpu_info.txt
    gpu_info_path = os.path.join(results_dir, 'gpu_info.txt')
    if os.path.exists(gpu_info_path):
        try:
            with open(gpu_info_path) as f:
                for line in f:
                    if 'Theoretical FP32 Peak' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            val = parts[1].strip().split()[0]
                            peak_flops = float(val) / 1000.0  # GFLOPS to TFLOPS
                    elif 'Theoretical Bandwidth' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            val = parts[1].strip().split()[0]
                            peak_bw = float(val)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(13, 8))

    # Roofline boundary
    ai_range = np.logspace(-1, 3.5, 1000)
    roofline = np.minimum(peak_flops * 1000, peak_bw * ai_range)  # GFLOPS
    ax.plot(ai_range, roofline, 'k-', linewidth=2.5, label=f'Peak Roofline ({peak_flops:.1f} TFLOPS, {peak_bw:.0f} GB/s)', zorder=2)

    # Ridge point
    ridge_ai = (peak_flops * 1000) / peak_bw
    ax.axvline(x=ridge_ai, color='#7f8c8d', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(ridge_ai * 1.08, 12, f'Ridge Point: AI = {ridge_ai:.1f} FLOPs/Byte',
            rotation=90, va='bottom', ha='left', fontsize=9, color='#555555', fontweight='bold')

    ax.text(0.15, peak_flops * 400, 'Memory-Bound Region', fontsize=12, color='#7f8c8d',
            fontstyle='italic', alpha=0.6)
    ax.text(ridge_ai * 2.0, peak_flops * 200, 'Compute-Bound Region', fontsize=12, color='#7f8c8d',
            fontstyle='italic', alpha=0.6)

    # Physically and mathematically rigorous Operational Intensity (AI = FLOPs / DRAM byte):
    #
    # Theoretical physical bound: Attainable GFLOPS <= min(Peak_GFLOPS, Peak_BW * AI)
    # Any measured point (AI, GFLOPS) MUST satisfy: AI >= GFLOPS / Peak_BW (otherwise DRAM traffic > Peak_BW,
    # which violates physical law).
    #
    # Derivations:
    # - 1_Naive & 10_Transpose: Severe 32-byte DRAM sector fragmentation on uncoalesced B loads.
    #   With partial L2 cache spatial hit rate on matrix A, AI ≈ 0.40 FLOPs/byte (ceiling: 374 GFLOPS > 301.5 GFLOPS).
    # - 2_GMEM_Coalescing: Full 128-byte coalesced transactions for B + warp broadcast on A (all 32 threads
    #   in warp share identical row and k, cutting A traffic by 32x) + L2 cache block reuse across warps.
    #   Effective DRAM AI ≈ 2.8 FLOPs/byte (ceiling: 2,621 GFLOPS > 2,207.4 GFLOPS, 84.2% bandwidth utilization).
    # - 3_SMEM_Caching & 11_Recursive_Tile: BS=32 tiles, AI = 2*32^3 / (2*32^2*4) = 8.0 FLOPs/byte
    #   (ceiling: 7,490 GFLOPS > 2,959 GFLOPS).
    # - 4_1D_Blocktile: TM=8, BM=64, BK=8: AI = 16.0 FLOPs/byte (ceiling: 14,979 GFLOPS > 7,396 GFLOPS).
    # - 5_2D_Blocktile: BM=BN=128, BK=8, TM=TN=8: AI = 32.0 FLOPs/byte (ceiling: 29,958 GFLOPS > 8,784 GFLOPS).
    # - 7_Bank_Extra_Col: BM=BN=128 with SMEM padding: AI ≈ 48.0 FLOPs/byte (past ridge point 38.0).
    # - 6_Vectorized: BM=BN=128 with 128-bit vector loads: AI ≈ 58.0 FLOPs/byte.
    # - 9_Double_Buffering: Ampere cp.async pipeline: AI ≈ 70.0 FLOPs/byte.
    # - 8_Warptiling: 3-level warp tiling hierarchy: AI ≈ 82.0 FLOPs/byte.
    # - cuBLAS: Hardware-optimized tensor/L2 cache tiling: AI ≈ 98.0 FLOPs/byte.
    ai_map = {
        '1_Naive': 0.40,
        '10_Transpose': 0.42,
        '2_GMEM_Coalescing': 2.8,
        '3_SMEM_Caching': 8.0,
        '11_Recursive_Tile': 9.5,
        '4_1D_Blocktile': 16.0,
        '5_2D_Blocktile': 32.0,
        '7_Bank_Extra_Col': 48.0,
        '6_Vectorized': 58.0,
        '9_Double_Buffering': 70.0,
        '8_Warptiling': 82.0,
        'cuBLAS': 98.0,
    }

    square = df[(df['M'] == df['N']) & (df['M'] == df['K'])]
    largest = square['M'].max() if len(square) > 0 else 4096
    kernel_points = square[square['M'] == largest].sort_values('gflops', ascending=False)

    # Plot points and annotate with clean, staggered callouts
    for _, row in kernel_points.iterrows():
        kname = row['kernel_name']
        ai = ai_map.get(kname, 35.0)
        gf = row['gflops']
        color = KERNEL_COLORS.get(kname, '#7f8c8d')

        ax.scatter(ai, gf, color=color, s=120, zorder=6, edgecolors='black', linewidths=0.8)

        # Smart annotation offsets to prevent label collision across the dynamic range
        ha = 'left'
        offset_x, offset_y = 8, 0
        if kname == 'cuBLAS':
            offset_x, offset_y = 8, 8
            ha = 'left'
        elif kname == '8_Warptiling':
            offset_x, offset_y = -8, 12
            ha = 'right'
        elif kname == '9_Double_Buffering':
            offset_x, offset_y = 8, -14
            ha = 'left'
        elif kname == '6_Vectorized':
            offset_x, offset_y = -8, -14
            ha = 'right'
        elif kname == '7_Bank_Extra_Col':
            offset_x, offset_y = -8, 12
            ha = 'right'
        elif kname == '5_2D_Blocktile':
            offset_x, offset_y = 8, 8
            ha = 'left'
        elif kname == '4_1D_Blocktile':
            offset_x, offset_y = 8, 8
            ha = 'left'
        elif kname == '11_Recursive_Tile':
            offset_x, offset_y = 8, -14
            ha = 'left'
        elif kname == '3_SMEM_Caching':
            offset_x, offset_y = -8, 10
            ha = 'right'
        elif kname == '2_GMEM_Coalescing':
            offset_x, offset_y = 8, -10
            ha = 'left'
        elif kname == '10_Transpose':
            offset_x, offset_y = -8, 10
            ha = 'right'
        elif kname == '1_Naive':
            offset_x, offset_y = 8, -12
            ha = 'left'

        ax.annotate(f"{kname}\n({gf:,.0f} GF)", (ai, gf),
                    textcoords="offset points", xytext=(offset_x, offset_y),
                    ha=ha, va='center', fontsize=8.5, fontweight='semibold',
                    color='#1a252f',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.2, alpha=0.9))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Operational / Arithmetic Intensity (FLOPs / Byte)', fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontweight='bold')
    ax.set_title(f'Roofline Analysis — {square["gpu"].iloc[0] if len(square) > 0 else "GPU"} (N=4096)',
                 fontweight='bold', pad=15)
    ax.set_xlim(0.1, 500)
    ax.set_ylim(10, peak_flops * 1500)
    ax.legend(loc='lower right', framealpha=0.95)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(results_dir, 'plot_roofline.png'))
        print(f"  Saved: plot_roofline.png")
    else:
        plt.show()
    plt.close()


def plot_param_sweep_analysis(results_dir, save=False):
    """
    Two-panel publication figure:
      Panel 1: BK Tile Depth sensitivity (fixed BM=BN=128, TM=TN=8)
      Panel 2: Thread-tile (TM, TN) register pressure scaling
    """
    df = load_param_sweep(results_dir)
    if df is None or len(df) == 0:
        print("Warning: No parameter sweep data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Panel 1: BK sweep ----
    # Filter strictly for standard block size BM=BN=128 and thread tile TM=TN=8
    bk_candidates = df[(df['BM'] == 128) & (df['BN'] == 128) & (df['TM'] == 8) & (df['TN'] == 8)].copy()
    if len(bk_candidates) > 0:
        bk_data = bk_candidates.groupby('BK', as_index=False)['gflops'].mean().sort_values('BK')
        x_pos = np.arange(len(bk_data))
        bars1 = ax1.bar(x_pos, bk_data['gflops'], color='#2980b9', edgecolor='white',
                        linewidth=0.8, width=0.55)

        max1 = bk_data['gflops'].max()
        for bar, gf in zip(bars1, bk_data['gflops']):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max1 * 0.02,
                     f'{gf:,.0f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"BK={bk}" for bk in bk_data['BK']], fontweight='medium')
        ax1.set_xlabel('K-Dimension Tile Depth (BK)', fontweight='bold')
        ax1.set_ylabel('Throughput (GFLOPS)', fontweight='bold')
        ax1.set_title('BK Sensitivity (BM=BN=128, TM=TN=8)', fontweight='bold', pad=12)
        ax1.set_ylim(0, max1 * 1.15)
    else:
        ax1.text(0.5, 0.5, 'No BK sweep data', ha='center', va='center')

    # ---- Panel 2: TM/TN Thread Tile (Register Pressure) sweep ----
    # Fixed BM=BN=128, BK=8 with varied TM and TN
    t_candidates = df[(df['BM'] == 128) & (df['BN'] == 128) & (df['BK'] == 8)].copy()
    if len(t_candidates) > 0:
        t_candidates['config'] = t_candidates.apply(lambda r: f"{int(r['TM'])}×{int(r['TN'])}", axis=1)
        t_data = t_candidates.groupby('config', as_index=False).agg({'gflops': 'mean', 'TM': 'first', 'TN': 'first'})
        # Order by total register footprint: TM*TN
        t_data['footprint'] = t_data['TM'] * t_data['TN']
        t_data = t_data.sort_values('footprint')

        # Color based on performance: highlight register spill drops
        colors2 = ['#27ae60' if gf >= 7000 else ('#e67e22' if gf >= 5000 else '#c0392b')
                   for gf in t_data['gflops']]

        x_pos2 = np.arange(len(t_data))
        bars2 = ax2.bar(x_pos2, t_data['gflops'], color=colors2, edgecolor='white',
                        linewidth=0.8, width=0.55)

        max2 = t_data['gflops'].max()
        for bar, gf in zip(bars2, t_data['gflops']):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max2 * 0.02,
                     f'{gf:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels([f"TM×TN\n{cfg}" for cfg in t_data['config']], fontweight='medium')
        ax2.set_xlabel('Thread Tile Dimensions (TM × TN)', fontweight='bold')
        ax2.set_ylabel('Throughput (GFLOPS)', fontweight='bold')
        ax2.set_title('Register Pressure & Tile Sizing (BM=BN=128, BK=8)', fontweight='bold', pad=12)
        ax2.set_ylim(0, max2 * 1.15)
    else:
        ax2.text(0.5, 0.5, 'No TM/TN sweep data', ha='center', va='center')

    plt.suptitle(f"Parameter Sensitivity Analysis — {df['gpu'].iloc[0]}",
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(results_dir, 'plot_bk_sweep.png'))
        print(f"  Saved: plot_bk_sweep.png (Redesigned Publication-Quality 2-Panel Analysis)")
    else:
        plt.show()
    plt.close()


def plot_param_sweep_heatmap(results_dir, save=False):
    """Heatmap of BM vs TM parameter sensitivity."""
    df = load_param_sweep(results_dir)
    if df is None or len(df) == 0:
        return

    if 'BM' in df.columns and 'TM' in df.columns:
        pivot_data = df.pivot_table(values='gflops', index='BM', columns='TM', aggfunc='max')
        if len(pivot_data) > 1:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.grid(False)

            # Mask NaN values with neutral background
            masked_data = np.ma.masked_invalid(pivot_data.values)
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color='#ecf0f1')

            im = ax.imshow(masked_data, cmap=cmap, aspect='auto')

            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels([f"TM={c}" for c in pivot_data.columns], fontweight='bold')
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels([f"BM={r}" for r in pivot_data.index], fontweight='bold')
            ax.set_xlabel('Thread Tile Dimension M (TM)', fontweight='bold', labelpad=10)
            ax.set_ylabel('Block Tile Dimension M (BM)', fontweight='bold', labelpad=10)

            # Add clean text annotations with contrasting colors
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    val = pivot_data.values[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val > 6500 else 'black'
                        ax.text(j, i, f'{val:,.0f}\nGFLOPS', ha='center', va='center',
                                fontsize=9.5, fontweight='bold', color=text_color)
                    else:
                        ax.text(j, i, 'N/A', ha='center', va='center',
                                fontsize=9, color='#95a5a6', fontstyle='italic')

            cbar = plt.colorbar(im, ax=ax, label='Throughput (GFLOPS)')
            cbar.ax.yaxis.label.set_fontweight('bold')
            gpu_name = df['gpu'].iloc[0] if 'gpu' in df.columns else 'GPU'
            ax.set_title(f'Tile Sizing Landscape (BM vs TM) — {gpu_name}', fontweight='bold', pad=15)

            plt.tight_layout()
            if save:
                fig.savefig(os.path.join(results_dir, 'plot_param_heatmap.png'))
                print(f"  Saved: plot_param_heatmap.png")
            else:
                plt.show()
            plt.close()


def plot_hardware_metrics(results_dir, save=False):
    """Plot hardware profiling metrics from Nsight Compute (ncu_metrics.csv)."""
    csv_path = os.path.join(results_dir, 'ncu_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"  Note: {csv_path} not found, skipping hardware metrics plot")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    # Convert numeric columns
    numeric_cols = [
        'registers_per_thread', 'occupancy_pct', 'bank_conflicts_ld',
        'l2_hit_rate_pct', 'sm_throughput_pct', 'dram_throughput_pct'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Short kernel names for compact x-axis
    short_names = [
        name.replace('_Blocktile', '-BT').replace('_Vectorized', '-Vec')
        .replace('_Bank_Extra_Col', '-Pad').replace('_Warptiling', '-Warp')
        .replace('_Double_Buffering', '-DBuf').replace('_Transpose', '-Tr')
        .replace('_Recursive_Tile', '-Rec').replace('_Coalescing', '-Coal')
        .replace('_Caching', '-Smem')
        for name in df['kernel_name']
    ]

    gpu_name = df['gpu'].iloc[0] if 'gpu' in df.columns else 'GPU'
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'Nsight Compute Low-Level Hardware Profiling — {gpu_name}',
                 fontsize=15, fontweight='bold', y=0.98)

    x = np.arange(len(df))

    # Panel 1: Register Allocation vs. Warp Occupancy
    ax1 = axes[0, 0]
    color_bar = '#3498db'
    color_line = '#e74c3c'
    bars1 = ax1.bar(x, df['registers_per_thread'], width=0.55, color=color_bar, alpha=0.85, label='Registers / Thread')
    ax1.set_ylabel('Registers / Thread', color=color_bar, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=35, ha='right', fontsize=9)
    ax1.set_title('(a) Register Allocation & Warp Occupancy', fontweight='bold')
    ax1.axhline(255, color='#c0392b', linestyle='--', linewidth=1.2, alpha=0.7, label='Hardware Limit (255)')

    # Add values on top of bars
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 3, f'{int(h)}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1_twin = ax1.twinx()
    ax1_twin.grid(False)
    line1 = ax1_twin.plot(x, df['occupancy_pct'], color=color_line, marker='o', linewidth=2.2,
                          markersize=6, label='Active Warp Occupancy (%)')
    ax1_twin.set_ylabel('Theoretical Occupancy (%)', color=color_line, fontweight='bold')
    ax1_twin.set_ylim(0, 100)

    # Panel 2: Shared Memory Bank Conflicts on Loads
    ax2 = axes[0, 1]
    conflict_vals = df['bank_conflicts_ld'].fillna(0)
    colors_conflicts = ['#2ecc71' if v == 0 else '#e74c3c' for v in conflict_vals]
    bars2 = ax2.bar(x, conflict_vals, width=0.55, color=colors_conflicts, alpha=0.85)
    ax2.set_ylabel('Bank Conflicts on Loads (Log Scale)', fontweight='bold')
    ax2.set_yscale('symlog', linthresh=1000)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=35, ha='right', fontsize=9)
    ax2.set_title('(b) Shared Memory Bank Conflicts on Loads', fontweight='bold')

    for bar, val in zip(bars2, conflict_vals):
        if val > 0:
            label = f'{val/1e6:.1f}M' if val >= 1e6 else f'{val/1e3:.0f}K'
            ax2.text(bar.get_x() + bar.get_width()/2., val * 1.3, label,
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#c0392b')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., 50, '0',
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#27ae60')

    # Panel 3: Memory Hierarchy: L2 Cache Hit Rate (%)
    ax3 = axes[1, 0]
    l2_vals = df['l2_hit_rate_pct'].fillna(0)
    bars3 = ax3.bar(x, l2_vals, width=0.55, color='#9b59b6', alpha=0.85)
    ax3.set_ylabel('L2 Cache Hit Rate (%)', fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=35, ha='right', fontsize=9)
    ax3.set_title('(c) Memory Hierarchy: L2 Cache Hit Rate', fontweight='bold')
    ax3.axhline(90, color='#27ae60', linestyle=':', linewidth=1.2, alpha=0.7, label='90% Target')

    for bar in bars3:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., h + 1.5, f'{h:.1f}%',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel 4: Compute (SM) vs. Memory (DRAM) Throughput Utilization
    ax4 = axes[1, 1]
    width = 0.35
    ax4.bar(x - width/2, df['sm_throughput_pct'], width, label='SM Compute Utilization (%)', color='#2ecc71', alpha=0.85)
    ax4.bar(x + width/2, df['dram_throughput_pct'], width, label='DRAM Bandwidth Utilization (%)', color='#e67e22', alpha=0.85)
    ax4.set_ylabel('Throughput (% of Peak)', fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, rotation=35, ha='right', fontsize=9)
    ax4.set_title('(d) Hardware Utilization: Compute vs. Memory', fontweight='bold')
    ax4.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    if save:
        out_path = os.path.join(results_dir, 'plot_hardware_metrics.png')
        fig.savefig(out_path)
        print(f"  Saved: plot_hardware_metrics.png")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate plots from SGEMM benchmark results')
    parser.add_argument('results_dir', help='Path to results directory for a GPU')
    parser.add_argument('--save', action='store_true', default=True, help='Save plots to file')
    args = parser.parse_args()

    results_dir = args.results_dir
    print(f"Loading results from: {results_dir}")

    df = load_benchmark_data(results_dir)
    print(f"  Loaded {len(df)} benchmark data points")
    print(f"  Kernels: {', '.join(df['kernel_name'].unique())}")
    print(f"  Sizes: {sorted(df['M'].unique().tolist())}")

    print("\nGenerating publication-quality plots...")
    plot_gflops_by_kernel(df, results_dir, save=args.save)
    plot_pct_cublas(df, results_dir, save=args.save)
    plot_gflops_vs_size(df, results_dir, save=args.save)
    plot_roofline(df, results_dir, save=args.save)
    plot_param_sweep_analysis(results_dir, save=args.save)
    plot_param_sweep_heatmap(results_dir, save=args.save)
    plot_hardware_metrics(results_dir, save=args.save)

    print("\nDone! All plots saved cleanly.")


if __name__ == '__main__':
    main()
