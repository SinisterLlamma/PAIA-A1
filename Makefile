# =============================================================================
# Makefile for CUDA SGEMM Performance Analysis
# =============================================================================

# Auto-detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ifeq ($(GPU_ARCH),)
  GPU_ARCH := 86
endif

NVCC       := nvcc
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_$(GPU_ARCH) --use_fast_math -lineinfo
LIBS       := -lcublas

SRC_DIR    := src
BUILD_DIR  := build
SCRIPTS    := scripts

# GPU name for results directory (auto-detected, spaces → underscores)
GPU_NAME   := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
RESULTS    := results/$(GPU_NAME)

.PHONY: all clean benchmark validate run sweep profile-ncu profile-nsys param-sweep gpu-info dirs

# =============================================================================
# Build targets
# =============================================================================
all: dirs $(BUILD_DIR)/benchmark $(BUILD_DIR)/validate

dirs:
	@mkdir -p $(BUILD_DIR) $(RESULTS)

$(BUILD_DIR)/benchmark: $(SRC_DIR)/benchmark.cu $(SRC_DIR)/kernels/*.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/validate: $(SRC_DIR)/validate.cu $(SRC_DIR)/kernels/*.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf $(BUILD_DIR)

# =============================================================================
# Run targets
# =============================================================================

# Validate all kernels for correctness
validate: $(BUILD_DIR)/validate
	@echo "===== Validating all kernels ====="
	$(BUILD_DIR)/validate

# Quick run with default sizes
run: $(BUILD_DIR)/benchmark
	@mkdir -p $(RESULTS)
	@echo "===== Quick benchmark (default sizes) ====="
	$(BUILD_DIR)/benchmark 2>$(RESULTS)/benchmark.log | tee $(RESULTS)/benchmarks.csv
	@echo "Results saved to $(RESULTS)/benchmarks.csv"

# Full dimension sweep
sweep: $(BUILD_DIR)/benchmark
	@mkdir -p $(RESULTS)
	@echo "===== Full dimension sweep ====="
	$(BUILD_DIR)/benchmark --sweep --runs 20 2>$(RESULTS)/sweep.log | tee $(RESULTS)/benchmarks.csv
	@echo "Results saved to $(RESULTS)/benchmarks.csv"

# Parameter sensitivity sweep
param-sweep: $(BUILD_DIR)/benchmark
	@mkdir -p $(RESULTS)
	@echo "===== Parameter sensitivity sweep ====="
	$(BUILD_DIR)/benchmark --sweep --param-sweep --runs 10 2>$(RESULTS)/param_sweep.log | tee $(RESULTS)/param_sweep.csv
	@echo "Results saved to $(RESULTS)/param_sweep.csv"

# =============================================================================
# Profiling targets
# =============================================================================

# Nsight Compute profiling
profile-ncu: $(BUILD_DIR)/benchmark
	@mkdir -p $(RESULTS)
	bash $(SCRIPTS)/run_ncu_profile.sh

# Nsight Systems profiling
profile-nsys: $(BUILD_DIR)/benchmark
	@mkdir -p $(RESULTS)/nsys_traces
	bash $(SCRIPTS)/run_nsys_profile.sh

# Collect GPU info
gpu-info:
	@mkdir -p $(RESULTS)
	bash $(SCRIPTS)/collect_gpu_info.sh

# =============================================================================
# Analysis targets
# =============================================================================
plots: $(RESULTS)/benchmarks.csv
	python3 analysis/plot_results.py $(RESULTS)

compare:
	python3 analysis/compare_gpus.py results/

# =============================================================================
# Convenience: do everything
# =============================================================================
full: validate gpu-info sweep param-sweep profile-ncu profile-nsys plots
	@echo "===== Full analysis complete ====="
	@echo "Results in: $(RESULTS)/"
