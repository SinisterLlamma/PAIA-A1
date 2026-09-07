# CUDA SGEMM Performance Analysis

Comprehensive performance analysis of CUDA matrix multiplication (GEMM) kernels, implementing progressive optimizations from naive to warp-tiled and double-buffered approaches. Based on and compared against [Simon Boehm's CUDA matmul optimization guide and repository](https://github.com/siboehm/SGEMM_CUDA).

---

## Performance Summary (NVIDIA GeForce RTX 3090, $N=4096$)

| Kernel ID | Kernel Name | Optimization Focus | Time (ms) | GFLOPS | % of cuBLAS |
|-----------|-------------|-------------------|-----------|--------|-------------|
| **0** | **cuBLAS (FP32)** | Vendor-optimized baseline | 5.64 ms | 24,384 GFLOPS | 100.0% |
| **1** | **1_Naive** | Baseline: 1 thread per element, 2D block | 455.87 ms | 301 GFLOPS | 1.2% |
| **2** | **2_GMEM_Coalescing** | 1D threadblock index remapping | 62.26 ms | 2,207 GFLOPS | 9.1% |
| **3** | **3_SMEM_Caching** | Shared memory cache-blocking ($32 \times 32$) | 46.44 ms | 2,959 GFLOPS | 12.1% |
| **4** | **4_1D_Blocktile** | Thread coarsening ($TM=8$), L2 column locality | 18.58 ms | 7,396 GFLOPS | 30.3% |
| **5** | **5_2D_Blocktile** | 2D register tiling ($TM=8, TN=8, BM=BN=128$) | 15.65 ms | 8,784 GFLOPS | 36.0% |
| **6** | **6_Vectorized** | 128-bit `float4` loads, transposed A in SMEM | 7.40 ms | 18,573 GFLOPS | 76.2% |
| **7** | **7_Bank_Extra_Col** | SMEM padding (`extraCols = 5`) for bank conflicts | 8.40 ms | 16,370 GFLOPS | 67.1% |
| **8** | **8_Warptiling** | 3-level Block $\to$ Warp $\to$ Sub-Warp $\to$ Thread | **6.28 ms** | **21,888 GFLOPS** | **89.8%** |
| **9** | **9_Double_Buffering** | Ampere `cp.async` (`cuda::memcpy_async` pipeline) | 7.59 ms | 18,105 GFLOPS | 74.2% |
| **10** | **10_Transpose** | Pre-transposing matrix B | 453.84 ms | 303 GFLOPS | 1.2% |
| **11** | **11_Recursive_Tile** | Hierarchical recursive cache-oblivious tiling | 45.96 ms | 2,991 GFLOPS | 12.3% |

---

## Comparison & Code Reuse from `siboehm/SGEMM_CUDA`

This codebase directly references and incorporates the cutting-edge optimization techniques from Simon Boehm's reference repository (`https://github.com/siboehm/SGEMM_CUDA`):

### What Was Reused & Adapted:
1. **L2 Spatial Locality Mapping (Kernel 4 & 5)**:
   - Boehm's observation: `cRow = blockIdx.y; cCol = blockIdx.x;` ensures blocks with sequential IDs access columns of $B$ sequentially while sharing rows of $A$, improving L2 cache hit rate by ~30%.
2. **128-bit Vectorization & Transposed Shared Memory (Kernel 6)**:
   - Reused Boehm's `float4` GMEM $\to$ SMEM transfers and transposed shared memory layout for matrix $A$ (`As[(innerColA * 4 + i) * BM + innerRowA] = tmp.x`).
3. **Shared Memory Bank Conflict Elimination (Kernel 7)**:
   - Reused Boehm's Kernel 8 padding technique (`const int extraCols = 5; __shared__ float Bs[BK * (BN + extraCols)]`).
4. **Warp-Level Tiling (Kernel 8)**:
   - Reused Boehm's 3-level tiling decomposition (`WMITER`, `WNITER`, `WSUBM`, `WSUBN`) and register fragment caching across sub-warp steps. Reaches **21,888 GFLOPS** on RTX 3090.
5. **Ampere Asynchronous Copy & Double Buffering (Kernel 9)**:
   - Reused Boehm's Kernel 12 implementation using Ampere hardware asynchronous memory copy (`cuda::memcpy_async` with `cuda::barrier`) to overlap tile $(k+1)$ GMEM transfers with tile $k$ computation.

### Architectural Enhancements Added in Our Codebase:
- **Universal Boundary Safety**: Boehm's code assumes matrices are strictly multiples of tile sizes and crashes with unaligned memory access on arbitrary dimensions. Our kernels include graceful boundary checking and fallback paths, passing **121/121 validation tests** across power-of-2, non-square (e.g. $1000 \times 500 \times 750$), and odd dimensions (e.g. $127 \times 127$, $513 \times 513$).
- **Modern Standards Compliance**: Cleaned shared memory barrier initialization (`alignas` storage) to eliminate CUDA compilation warnings on modern NVCC toolchains.
- **Unified Benchmarking & Validation Suite**: Command-line driven execution, automated error checking against cuBLAS, CSV telemetry, and multi-GPU collection pipelines.

---

> 📄 **Detailed Preliminary Report**: See [report/preliminary_report.md](report/preliminary_report.md) for the complete microarchitectural analysis, Simon Boehm repository comparison, roofline model derivations, and empirical scaling evaluation.

---

## Project Structure

```
├── src/
│   ├── kernels/
│   │   ├── kernel_common.cuh           # Shared macros, timers, matrix helpers, enums
│   │   ├── kernel_1_naive.cuh          # K1: Naive (1 thread → 1 output element)
│   │   ├── kernel_2_coalescing.cuh     # K2: Global memory coalescing
│   │   ├── kernel_3_smem.cuh           # K3: Shared memory tiling (32x32)
│   │   ├── kernel_4_1d_blocktile.cuh   # K4: 1D blocktiling (TM=8, L2 locality)
│   │   ├── kernel_5_2d_blocktile.cuh   # K5: 2D blocktiling (TM=8, TN=8, BM=BN=128)
│   │   ├── kernel_6_vectorized.cuh     # K6: Vectorized float4 & transposed SMEM
│   │   ├── kernel_7_bank_conflicts.cuh # K7: Bank conflict resolution via padding
│   │   ├── kernel_8_warptiling.cuh     # K8: Warp-level tiling (sub-warp tiles)
│   │   ├── kernel_9_double_buffering.cuh # K9: Ampere cp.async double buffering
│   │   ├── kernel_cublas.cuh           # K0: cuBLAS reference baseline
│   │   └── kernel_extra.cuh            # K10/K11: Transpose & recursive tiling
│   ├── benchmark.cu                    # Main benchmark CLI driver
│   └── validate.cu                     # Correctness validation vs cuBLAS (121 tests)
├── scripts/
│   ├── run_benchmarks.sh               # Full benchmark sweep
│   ├── run_ncu_profile.sh              # Nsight Compute hardware profiling
│   ├── run_nsys_profile.sh             # Nsight Systems timeline profiling
│   ├── run_param_sweep.sh              # Parameter sensitivity sweep
│   └── collect_gpu_info.sh             # Hardware specs collection
├── analysis/
│   ├── plot_results.py                 # Single-GPU visualization & roofline model
│   └── compare_gpus.py                 # Cross-GPU comparative plotting
├── results/                            # Results stored per GPU (auto-detected)
│   └── NVIDIA_GeForce_RTX_3090/
│       ├── gpu_info.txt
│       ├── benchmarks.csv
│       ├── param_sweep.csv
│       ├── plot_gflops_by_kernel.png
│       ├── plot_pct_cublas.png
│       ├── plot_gflops_vs_size.png
│       ├── plot_roofline.png
│       ├── plot_bk_sweep.png
│       └── plot_param_heatmap.png
├── report/
│   ├── preliminary_report.md           # Comprehensive technical report
│   └── assets/                         # High-res microarchitecture figures
├── Makefile
└── README.md
```

---

## How to Run on Any GPU

When cloning this repo onto a different GPU machine:

```bash
# 1. Build everything (auto-detects compute capability: sm_80, sm_86, sm_89, sm_90, etc.)
make all

# 2. Validate correctness (runs all 11 kernels across 11 test configurations)
make validate

# 3. Collect hardware specifications (SM count, bandwidth, cache sizes)
make gpu-info

# 4. Run benchmarks
make run       # Default square sizes (1024, 2048, 4096)
make sweep     # Full dimension sweep (including non-square matrices)

# 5. Run parameter sensitivity sweep
bash scripts/run_param_sweep.sh

# 6. Profile with Nsight Compute (hardware counters)
make profile-ncu

# 7. Profile with Nsight Systems (timeline traces)
make profile-nsys

# 8. Generate visual plots
make plots

# 9. Commit results for cross-GPU comparison
git add results/<GPU_NAME>/
git commit -m "Add benchmark results for <GPU_NAME>"
```

---

## Nsight Compute (`ncu`) vs Nsight Systems (`nsys`)

- **Nsight Systems (`nsys`)**:
  - Use for system-level timeline analysis, API call overhead, kernel launch latencies, and host-device memory transfers.
  - Generates `.nsys-rep` trace files openable in the Nsight Systems GUI.
  - Run with: `bash scripts/run_nsys_profile.sh`.

- **Nsight Compute (`ncu`)**:
  - Use for deep kernel microarchitecture profiling: shared memory bank conflicts, compute vs memory throughput roofline, register spills, warp occupancy, and cache hit rates.
  - Generates `.ncu-rep` reports and CSV metrics tables.
  - Run with: `bash scripts/run_ncu_profile.sh`.

---

## Multi-GPU Cross Comparison

When results from multiple GPUs are committed to the `results/` directory (e.g. `results/NVIDIA_GeForce_RTX_3090/`, `results/NVIDIA_A100-SXM4-40GB/`, `results/NVIDIA_H100_80GB_HBM3/`), run:

```bash
python3 analysis/compare_gpus.py results/
```

This generates:
- Cross-GPU peak GFLOPS comparisons
- Scaling efficiency across architectures
- Relative efficiency as a percentage of cuBLAS
- Arithmetic intensity and roofline comparisons
