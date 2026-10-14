# Makefile for the sparse Hermitian / general eigensolver prototype.
#
# The system `python3` on this machine is 3.9.6 and cannot install the package
# (pyproject requires >= 3.10). Use a 3.10+ interpreter. Override with:
#   make PYTHON=/path/to/python3.11
PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
SRC    := src
export PYTHONPATH := $(SRC)

# Representative public matrices (see data/README.md). Not all may be present.
MATRIX ?= data/raw/CEMW__vfem.npz
SIGMA  ?= 0.1
K      ?= 6
OUT    ?= results/eigenpairs.npz
SLEPC_MPICC ?= mpicc
SLEPC_PKG ?= slepc-dmo
SLEPC_PKG_CONFIG ?= pkg-config
SLEPC_CFLAGS ?= $(shell $(SLEPC_PKG_CONFIG) --cflags $(SLEPC_PKG))
SLEPC_LIBS ?= $(shell $(SLEPC_PKG_CONFIG) --libs $(SLEPC_PKG))
ifeq ($(OS),Windows_NT)
SLEPC_EXE_SUFFIX ?= .exe
endif

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: check-python
check-python: ## Verify the interpreter is >= 3.10
	@$(PYTHON) -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10, got %s" % sys.version.split()[0]; print("python", sys.version.split()[0], "OK")'

.PHONY: install
install: check-python ## Editable install of the package (runtime deps only)
	$(PIP) install -e .

.PHONY: dev
dev: check-python ## Editable install with test extras
	$(PIP) install -e '.[test]'

.PHONY: test
test: ## Run the unit test suite
	$(PYTEST) -q

.PHONY: test-verbose
test-verbose: ## Run the unit test suite with verbose output
	$(PYTEST) -v

.PHONY: lint
lint: ## Byte-compile the package and tools (syntax check)
	$(PYTHON) -m compileall -q $(SRC) tools && echo "compileall OK"

.PHONY: run
run: ## Solve one matrix: make run MATRIX=... SIGMA=... K=...
	$(PYTHON) -m sparse_eigensolver "$(MATRIX)" --output "$(OUT)" -k $(K) --sigma $(SIGMA)

.PHONY: bench
bench: ## Run a synthetic benchmark: make bench N=10000 CASE=indefinite
	$(PYTHON) -m sparse_eigensolver.benchmark --n $(or $(N),10000) --case $(or $(CASE),indefinite) \
		--output results/benchmark.json

.PHONY: fetch-data
fetch-data: ## Download the public electromagnetic matrices into data/raw
	$(PYTHON) tools/fetch_matrices.py --out data/raw

.PHONY: strategies
strategies: ## Rebuild results/STRATEGY_COMPARISON.md (fill-in strategies)
	$(PYTHON) tools/benchmark_strategies.py

.PHONY: public-suite
public-suite: ## Run every downloaded matrix through both code paths
	$(PYTHON) tools/run_public_suite.py

.PHONY: fault-scenarios
fault-scenarios: ## Run the fault-scenario suite -> results/fault_scenarios.json
	$(PYTHON) -W ignore tools/run_fault_scenarios.py --timeout 700

.PHONY: fault-report
fault-report: ## Render results/FAULT_SCENARIOS.md from the JSON records
	$(PYTHON) tools/render_fault_report.py

.PHONY: accuracy
accuracy: ## Build the accuracy report (subspace angle / eigenvalue error)
	$(PYTHON) -W ignore tools/accuracy_report.py

.PHONY: adaptive-ilu
adaptive-ilu: ## Run the deterministic adaptive-ILU benchmark -> results/adaptive_ilu.json
	$(PYTHON) tools/benchmark_adaptive_ilu.py

.PHONY: speedup
speedup: ## Measure BLAS/OpenMP thread scaling -> results/SPEEDUP.md
	$(PYTHON) -W ignore tools/speedup_benchmark.py --case memory_limited --n $(or $(N),90000) --threads 1 2 4 8

.PHONY: capabilities
capabilities: ## Record integrated CPU/MPI/GPU capability status
	$(PYTHON) tools/check_parallel_capabilities.py

.PHONY: gpu-run
gpu-run: ## Solve one Hermitian matrix on CUDA; set MATRIX, OUT, SIGMA, K
	$(PYTHON) -m sparse_eigensolver $(MATRIX) --output $(OUT) --backend gpu --mode hermitian -k $(K) --sigma $(SIGMA)

.PHONY: slepc-mpi
slepc-mpi: ## Solve one matrix with PETSc/SLEPc distributed MPI; set MPIEXEC, RANKS, MATRIX, OUT
	$(or $(MPIEXEC),mpiexec) -n $(or $(RANKS),2) $(PYTHON) tools/run_slepc.py $(MATRIX) --output $(OUT) -k $(K) --sigma $(SIGMA)

.PHONY: slepc-native-build
slepc-native-build: ## Build the native PETSc/SLEPc MPI probe; set SLEPC_PKG if needed
	$(SLEPC_MPICC) -std=c11 -O2 tools/run_slepc_native.c -o tools/run_slepc_native$(SLEPC_EXE_SUFFIX) \
		$(SLEPC_CFLAGS) $(SLEPC_LIBS)

.PHONY: slepc-native-mpi
slepc-native-mpi: ## Run native PETSc/SLEPc MPI probe; set MATRIX to .mtx and OUT to .json
	$(or $(MPIEXEC),mpiexec) -n $(or $(RANKS),2) tools/run_slepc_native$(SLEPC_EXE_SUFFIX) \
		$(MATRIX) $(or $(OUT),results/slepc_native.json) $(K) $(SIGMA) 1e-10

.PHONY: mpi-batch
mpi-batch: ## Run independent solve jobs with MPI ranks; set MPIEXEC and JOBS
	$(or $(MPIEXEC),mpiexec) -n $(or $(RANKS),2) $(PYTHON) tools/run_mpi_batch.py \
		--jobs $(or $(JOBS),configs/mpi_batch_example.json) --output results/mpi_batch.json

.PHONY: reports
reports: ## Regenerate all result artefacts that do not need the network
	$(PYTHON) tools/benchmark_strategies.py
	$(PYTHON) -W ignore tools/run_fault_scenarios.py --timeout 700
	$(PYTHON) tools/render_fault_report.py
	$(PYTHON) -W ignore tools/accuracy_report.py
	$(PYTHON) -W ignore tools/speedup_benchmark.py --case memory_limited --n $(or $(N),90000) --threads 1 2 4 8

.PHONY: clean
clean: ## Remove caches and build artefacts (keeps results/ and data/)
	rm -rf build dist *.egg-info .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete

.PHONY: distclean
distclean: clean ## Also remove generated results and downloaded data
	rm -rf results/*.npz results/*.json data/raw


