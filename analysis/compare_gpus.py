#!/usr/bin/env python3
"""
compare_gpus.py – Cross-GPU comparison plots.

Reads results from multiple GPU directories under results/ and generates
comparison charts.

Usage:
    python3 analysis/compare_gpus.py results/
    python3 analysis/compare_gpus.py results/ --save
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

GPU_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
    '#1abc9c', '#e67e22', '#2c3e50',
]


def find_gpu_results(results_root):
    """Find all GPU result directories."""
    gpus = []
    for entry in sorted(os.listdir(results_root)):
        csv_path = os.path.join(results_root, entry, 'benchmarks.csv')
        if os.path.isdir(os.path.join(results_root, entry)) and os.path.exists(csv_path):
            gpus.append(entry)
    return gpus


def load_all_results(results_root, gpu_dirs):
    """Load and concatenate results from all GPUs."""
    frames = []
    for gpu in gpu_dirs:
        csv_path = os.path.join(results_root, gpu, 'benchmarks.csv')
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['kernel_name', 'gflops'])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_gpu_comparison_by_kernel(df, gpu_dirs, results_root, save=False):
    """Side-by-side bar chart comparing GFLOPS across GPUs for each kernel."""
    # Use largest common square size
    square = df[(df['M'] == df['N']) & (df['M'] == df['K'])]
    common_sizes = set(square[square['gpu'] == gpu_dirs[0].replace(' ', '_')]['M'].unique())
    for gpu in gpu_dirs[1:]:
        gpu_clean = gpu.replace(' ', '_')
        common_sizes &= set(square[square['gpu'] == gpu_clean]['M'].unique())

    if not common_sizes:
        print("Warning: No common matrix sizes across GPUs")
        return

    size = max(common_sizes)
    data = square[square['M'] == size]

    kernels = sorted(data['kernel_name'].unique(),
                     key=lambda x: data[data['kernel_name'] == x]['kernel_id'].min())
    gpu_names = sorted(data['gpu'].unique())

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(kernels))
    width = 0.8 / len(gpu_names)

    for i, gpu in enumerate(gpu_names):
        gpu_data = data[data['gpu'] == gpu]
        gflops = []
        for k in kernels:
            val = gpu_data[gpu_data['kernel_name'] == k]['gflops']
            gflops.append(val.values[0] if len(val) > 0 else 0)

        bars = ax.bar(x + i * width, gflops, width,
                      label=gpu.replace('_', ' '),
                      color=GPU_COLORS[i % len(GPU_COLORS)],
                      edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + width * len(gpu_names) / 2)
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    ax.set_ylabel('GFLOPS')
    ax.set_title(f'Cross-GPU Performance Comparison ({size}×{size} SGEMM)')
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(results_root, 'plot_gpu_comparison.png'))
        print(f"  Saved: plot_gpu_comparison.png")
    else:
        plt.show()
    plt.close()


def plot_gpu_scaling(df, gpu_dirs, results_root, save=False):
    """Line plot: GFLOPS vs matrix size for each GPU (cuBLAS only)."""
    square = df[(df['M'] == df['N']) & (df['M'] == df['K'])]
    cublas = square[square['kernel_name'] == 'cuBLAS']

    if len(cublas) == 0:
        print("Warning: No cuBLAS data for GPU scaling plot")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, gpu in enumerate(sorted(cublas['gpu'].unique())):
        gpu_data = cublas[cublas['gpu'] == gpu].sort_values('M')
        ax.plot(gpu_data['M'], gpu_data['gflops'], 'o-',
                color=GPU_COLORS[i % len(GPU_COLORS)],
                label=gpu.replace('_', ' '), linewidth=2, markersize=8)

    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('GFLOPS (cuBLAS)')
    ax.set_title('cuBLAS Performance Scaling Across GPUs')
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(results_root, 'plot_gpu_scaling.png'))
        print(f"  Saved: plot_gpu_scaling.png")
    else:
        plt.show()
    plt.close()


def plot_efficiency_comparison(df, gpu_dirs, results_root, save=False):
    """Bar chart: % of cuBLAS per kernel, grouped by GPU."""
    square = df[(df['M'] == df['N']) & (df['M'] == df['K'])]

    # Find common largest size
    common_sizes = None
    for gpu in df['gpu'].unique():
        sizes = set(square[square['gpu'] == gpu]['M'].unique())
        common_sizes = sizes if common_sizes is None else common_sizes & sizes

    if not common_sizes:
        return

    size = max(common_sizes)
    data = square[square['M'] == size].copy()

    fig, ax = plt.subplots(figsize=(14, 7))

    gpu_names = sorted(data['gpu'].unique())
    # Exclude cuBLAS from the comparison
    kernels_no_cublas = [k for k in sorted(data['kernel_name'].unique())
                         if k != 'cuBLAS']

    x = np.arange(len(kernels_no_cublas))
    width = 0.8 / len(gpu_names)

    for i, gpu in enumerate(gpu_names):
        gpu_data = data[data['gpu'] == gpu]
        cublas_gf = gpu_data[gpu_data['kernel_name'] == 'cuBLAS']['gflops']
        if len(cublas_gf) == 0:
            continue
        cublas_val = cublas_gf.values[0]

        pcts = []
        for k in kernels_no_cublas:
            val = gpu_data[gpu_data['kernel_name'] == k]['gflops']
            pct = (val.values[0] / cublas_val * 100) if len(val) > 0 else 0
            pcts.append(pct)

        ax.bar(x + i * width, pcts, width,
               label=gpu.replace('_', ' '),
               color=GPU_COLORS[i % len(GPU_COLORS)])

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x + width * len(gpu_names) / 2)
    ax.set_xticklabels(kernels_no_cublas, rotation=45, ha='right')
    ax.set_ylabel('% of cuBLAS')
    ax.set_title(f'Kernel Efficiency Across GPUs ({size}×{size})')
    ax.legend()
    ax.set_ylim(0, 110)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(results_root, 'plot_efficiency_comparison.png'))
        print(f"  Saved: plot_efficiency_comparison.png")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare SGEMM results across GPUs')
    parser.add_argument('results_root', help='Path to results/ directory')
    parser.add_argument('--save', action='store_true', help='Save plots as PNGs')
    args = parser.parse_args()

    gpu_dirs = find_gpu_results(args.results_root)
    if len(gpu_dirs) == 0:
        print(f"No GPU results found in {args.results_root}")
        print("Run benchmarks first: make sweep")
        sys.exit(1)

    print(f"Found results for {len(gpu_dirs)} GPU(s):")
    for g in gpu_dirs:
        print(f"  - {g}")

    if len(gpu_dirs) < 2:
        print("\nNote: Cross-GPU comparison requires results from at least 2 GPUs.")
        print("Run benchmarks on another GPU and copy results here.")
        print("Generating single-GPU summary instead...")

    df = load_all_results(args.results_root, gpu_dirs)
    print(f"\nTotal data points: {len(df)}")

    print("\nGenerating comparison plots...")
    plot_gpu_comparison_by_kernel(df, gpu_dirs, args.results_root, save=args.save)
    plot_gpu_scaling(df, gpu_dirs, args.results_root, save=args.save)
    plot_efficiency_comparison(df, gpu_dirs, args.results_root, save=args.save)

    print("\nDone!")


if __name__ == '__main__':
    main()
