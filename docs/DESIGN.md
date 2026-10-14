# 方案说明文档 (Design and method)

面向电磁谐振与微波器件仿真的高效稳定稀疏特征值求解方案。本文档说明**算法原理、复杂度与内存分析、收敛性、预处理策略、并行方案、评测结果、求解边界，以及电磁仿真的诊断流程**。配套的降级/容错说明见 [`DEGRADED_MODE.md`](DEGRADED_MODE.md)，精度数据见 [`../results/ACCURACY.md`](../results/ACCURACY.md)，故障场景见 [`../results/FAULT_SCENARIOS.md`](../results/FAULT_SCENARIOS.md)，并行加速见 [`../results/SPEEDUP.md`](../results/SPEEDUP.md)。

## 1. 问题定义与物理背景

电磁谐振/本征模仿真（腔体、波导、微波器件）经有限元/有限差分离散后，得到广义或标准特征值问题

```
A x = λ x        （标准型，本方案直接支持）
A x = λ B x      （广义型，B 对称正定，支持 `--mass-matrix`）
```

其中 `A` 为大型稀疏矩阵（`n` 可达 10^5–10^6），只关心**目标频段附近**的少数（`k` 个）特征对。物理上有两类来源：

- **无耗/理想导体**：`A` 为（复）Hermitian，`A = A^H`，特征值实。
- **含损耗/PML/辐射边界**：`A` 为复对称但**非 Hermitian**，`A = A^T ≠ A^H`，特征值可带虚部。

**这是正确性问题而非风格问题**：同一矩阵走 Hermitian 路径与 general 路径会得到不同谱。方案两条路径都实现，并在 `--mode auto` 下按相对非对称度 `||A - A^H||/||A||` 自动判定（阈值 `--hermitian-tol`，默认 `1e-12`）。

## 2. 算法原理：Shift-invert Lanczos / Arnoldi

直接对 `A` 做 Krylov 迭代求内部特征值收敛极慢。本方案用 **shift-invert（移位求逆）**：给定目标移位 `σ`，对算子

```
OP = (A - σ I)^{-1}
```

做 Krylov 迭代。`A - σ I` 的特征值 `μ = λ - σ` 映射为 `OP` 的特征值 `1/μ`：靠近 `σ` 的特征值 `μ→0`，其倒数**模最大**，因此可被 Lanczos/Arnoldi 快速收敛捕捉。

- Hermitian 路径：`scipy.sparse.linalg.eigsh(A, k, sigma=σ, mode="normal")` → 隐式重启动 Lanczos。
- General 路径：`scipy.sparse.linalg.eigs(A, k, sigma=σ)` → 隐式重启动 Arnoldi。

**每次外层迭代**需要一次 `(A - σ I)^{-1} v` 求解（一次线性方程组求解），这是全部成本所在（见 §3）。

**Hermitian 化**：auto 模式判定为 Hermitian 时，对 `(A + A^H)/2` 求解以消除舍入造成的微小非对称，再对**原始矩阵**校验残差。

**三角恢复（按需）**：部分电磁软件只存储三角（对称/Hermitian 存储）。`--triangular {none,auto,hermitian,symmetric}` 可据存储的三角重建完整矩阵：`hermitian` 镜像共轭（`tril(A,0) + tril(A,-1)^H`），`symmetric` 镜像不共轭（`tril(A,0) + tril(A,-1)^T`）。共轭与否正是 Hermitian 与复对称的分界。`auto` 只能依据**对角元**判断：Hermitian 矩阵对角为实，复对称（含损耗/PML）矩阵对角一般带虚部——故实矩阵或实对角 → Hermitian 镜像，复对角 → 对称镜像。约定已知时应显式指定 `hermitian`/`symmetric`。

## 3. 复杂度与内存分析

设 `nnz` 为 `A` 的非零元个数，`nnz(L+U)` 为 `A - σ I` 稀疏 LU 分解的非零元个数（填充 fill-in）。

| 阶段 | 时间复杂度 | 内存 |
|---|---|---|
| 预处理（重排 RCM） | `O(nnz)` | `O(nnz)` |
| 稀疏 LU 分解 | `O(fill × 列数)`，实测远大于 `O(nnz)` | `O(nnz(L+U))` |
| 每次内解（三角回代） | `O(nnz(L+U))` | `O(n)` |
| 每次 ARPACK 外层迭代 | 一次内解 + `O(n·ncv)` | `O(n·ncv)` |
| 单次 ILU+GMRES 内解 | `O(gmres_iter × nnz(L+U_ilu))` | `O(nnz(L+U_ilu))` |

**关键结论**：

1. **成本由填充而非 `n` 决定**。对 3-D 电磁问题，LU 填充可达 `nnz(A)` 的数十到上千倍。实测：`Um__2cubes_sphere`（n=101k）精确 LU 填充 247×（~406M 非零，~5.5 min，~4.1 GB）；`CEMW__vfem`（n=93k，复）148×（~212M 非零，~500 s）。
2. **总时间 ≈ `(外层迭代数) × (一次内解成本)`**。因此降低内解成本（重排、ILU）是首要杠杆。
3. **峰值内存 ≈ `O(nnz(L+U))`**，与线程数无关（并行不改变内存预算）。
4. ARPACK 内存随 `n × ncv` 增长；`--ncv` 是可控的次要杠杆。

### 峰值内存与策略选择（指标 4-性能）

求解器在每次运行的诊断 JSON 中记录 `peak_rss_mb`（进程峰值常驻内存，macOS 为字节、Linux 为 KB，统一换算为 MB）。决策流程：

```
估算 nnz(A) 与目标 σ
      │
      ├─ 试跑：--inner-solver direct，记录 peak_rss_mb 与 factor_nnz
      │
      ├─ 若 peak_rss_mb 超出预算（如 16 GB 机器的安全线 ~12 GB）：
      │      1. --reorder rcm  （免费降填充，谱不变）
      │      2. --permc-spec MMD_AT_PLUS_A  （换 SuperLU 列序）
      │      3. --inner-solver ilu --ilu-fill-factor 20  （精确 LU → ILU+GMRES）
      │      4. 仍超预算：--ilu-drop-tol 1e-2 … 1e-1，或减小 -k / --ncv
      │
      └─ 每次改动后复查 peak_rss_mb 与残差，二者都必须满足
```

- **硬内存预算**：Linux 用 `ulimit -v`/`RLIMIT_AS`；macOS 无法设置有限 `RLIMIT_AS`，改用父进程监控子进程 RSS 并在越界时终止（见 `tools/run_fault_scenarios.py` 的 `memory_limited` 场景与 [`DEGRADED_MODE.md`](DEGRADED_MODE.md)）。
- 内存受限场景（3-D 7 点 Laplacian，n=64,000，1 GB 预算）：精确 LU 填充 ~1500×（~1.7 GB）被杀；ILU（fill-factor 20）仅 ~0.29 GB、残差 ~5e-12，在预算内完成。

## 4. 收敛性分析

- **Lanczos/Arnoldi 收敛速率**由目标特征值与其余谱的间隔决定。Shift-invert 把"最靠近 `σ` 的特征值"变换为"模最大的特征值"，间隔被放大为 `|1/μ_i - 1/μ_j|`，故收敛快。
- **`ncv`（Krylov 子空间维数）**默认 `min(n, max(2k+1, 20))`；`ncv` 越大每步成本越高但迭代数越少，需要权衡。
- **退化/重根特征值**：多重特征值是 Krylov 方法的固有弱点——同一特征值的多个特征向量张成的子空间收敛慢，且**可能只返回部分重根向量**。方案对策：
  - 请求略多于预期重数的 `k`，并校验返回向量的**正交性**（诊断记录 `orthogonality_error_fro`，见 `results/ACCURACY.md`）。
  - 用 `--shift-probes` 在多个近邻移位探测，避免单一移位漏掉简并簇。
  - `repeated` / `clustered` / `near_shift` 故障场景（见 `results/FAULT_SCENARIOS.md`）专门覆盖这些退化情形。
- **近奇异移位**：`σ` 落在特征值上会使 `A - σ I` 奇异。方案按候选移位阶梯重试：`--shift-probes`（绝对偏移，按序尝试）与 `--adaptive-shift`（几何回退 `step, 2·step, 5·step, 10·step, 20·step, 50·step, 100·step`，双符号），并用 `--target-band` 从频带中心向外走。实际使用的移位记录在 `effective_shift`。
- **低频频变（low-frequency breakdown）**：零附近聚集大量特征值，间隔趋零、收敛退化；场景 `low_freq_breakdown` 专门覆盖。

## 5. 预处理策略（降填充 / 提速）

Shift-invert 分解 `A - σ I`。三个**独立且保持谱不变**的旋钮：

1. **重排 `--reorder rcm`**：对 `A` 做反 Cuthill–McKee 置换，分解 `P A P^T` 并映射回特征向量，残差始终对原矩阵校验。
2. **SuperLU 列序 `--permc-spec {NATURAL,MMD_AT_PLUS_A,MMD_ATA,COLAMD}`**：默认 `COLAMD`；`MMD_*` 在部分矩阵上填充更小。
3. **ILU + GMRES `--inner-solver ilu`**：用不完全 LU（`--ilu-drop-tol`、`--ilu-fill-factor`）替代精确 LU，内解改为 GMRES（`--gmres-restart`、`--gmres-rtol`）。内存显著降低，代价是内解迭代数上升。

实测（填充比 = 分解 nnz / nnz(A)）：`Oberwolfach__gyro`（n=17k）`MMD_AT_PLUS_A` 3.05× vs `COLAMD` 6.63× vs `NATURAL` 24.72×；`Um__2cubes_sphere` ILU 把 247× 降到 18.8×；`CEMW__vfem` ILU 把 148× 降到 1.56×。详见 [`../results/STRATEGY_COMPARISON.md`](../results/STRATEGY_COMPARISON.md)。

> 经验：列序选择**依矩阵而异**，上线前先做廉价试跑；**绝不**把 `NATURAL` 作为默认。若 ILU 报 `Factor is exactly singular`，通常是丢弃容差导致的零主元（非移位奇异），先提高 `--ilu-fill-factor`（10→20→50）。

## 6. 并行方案（指标 2f / 4-性能）

- **单矩阵本原型为单进程 CPU**，并行性来自 BLAS/LAPACK 多线程（OpenBLAS/MKL），覆盖稠密核（分解中的密集块、GMRES 内积）与部分稀疏核；`tools/run_mpi_batch.py` 另提供多个独立任务的 MPI 批处理。
- **线程控制**：`--threads N` 调用 `set_threads(N)`，通过 `threadpoolctl` 限制已加载的 BLAS 线程池，并导出 `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` 供子进程使用；`set_threads(0)` 表示不限制。
- **是否有效**：由 `tools/speedup_benchmark.py` 量化（`make speedup` → `results/SPEEDUP.md`），在 1/2/4/8 线程下测端到端墙钟、加速比 `S(p)=T(1)/T(p)` 与效率 `S(p)/p`。
- **实测结论（重要）**：在本工作负载上**并行加速基本无效**——最好成绩出现在 `p=1`，线程越多反而略慢（稀疏 LU 分解与三角回代是内存带宽受限且高度串行，BLAS 多线程帮不上忙，同步/调度开销随线程数上升）。因此对**稀疏**特征值问题，加速应来自**算法**（重排/ILU 降填充）而非线程数；线程旋钮的价值在于**可复现性**（固定 `--threads 1`）与**避免超订**。
- **峰值内存与线程数基本无关**，并行不改变内存预算结论。
- **并行后端**：默认单矩阵路径仍是 SciPy CPU；GPU Hermitian 路径通过 `--backend gpu` 接入 CuPy/CUDA。真正的单矩阵 MPI 路径通过可选 `tools/run_slepc.py` 接入 PETSc/SLEPc；只有在匹配的 MPI/PETSc/SLEPc 运行时安装后才可启用。独立任务批处理仍由 `tools/run_mpi_batch.py` 提供。

## 7. 评测结果汇总

| 维度 | 报告 | 要点 |
|---|---|---|
| 精度（子空间夹角/特征值误差） | [`../results/ACCURACY.md`](../results/ACCURACY.md) | 小规模对照稠密 `eigh`/`eig`；大规模 LU vs ILU 互校，夹角达机器精度量级 |
| 迭代次数 | 同上 + 故障表 | `arpack_iterations`（外层）、`inner_matvecs`（内解/`(A-σI)` 应用） |
| 性能/内存 | [`../results/STRATEGY_COMPARISON.md`](../results/STRATEGY_COMPARISON.md) | 填充比、时间、峰值 RSS；重排/ILU 的降填充效果 |
| 并行加速 | [`../results/SPEEDUP.md`](../results/SPEEDUP.md) | 线程数 → 墙钟/加速比/效率/峰值内存 |
| 故障场景 | [`../results/FAULT_SCENARIOS.md`](../results/FAULT_SCENARIOS.md) | 每场景的精度/性能/迭代/处置建议 |

## 8. 求解边界（明确不承诺的部分）

- **广义问题 `A x = λ B x`**：支持稀疏 `B` 输入。shift-invert 直接因子化 `(A - σB)`，并将 `B` 作为 ARPACK 的 `M` 算子；残差按 `A x - λB x` 在原始矩阵上验证。Hermitian 路径要求 `B` 为 Hermitian，并遵守 SciPy 对广义问题的正定/半正定约束。
- **Hermitian vs 复对称的正确约定**：必须与主办方确认。同一 `FEMLAB/waveguide3D`，general 路径给出 `[-3.8e-4+2.1e-6j, ...]`，强制 Hermitian 给出全实 `[-1.8e-4, ...]`，**两谱不重叠**。
- **不返回未请求的特征值**：有限 `k` 无法证明没有更靠近 `σ` 的特征值未被返回。
- **`eigsh` 要求 `1 ≤ k < n`**；接近满谱或极小矩阵需单独的稠密流程。
- **后端边界**：GPU Hermitian 求解通过 CuPy/CUDA；单矩阵 MPI 通过可选 PETSc/SLEPc 入口提供，当前是否可用由 `make capabilities` 检测；独立任务可通过 MPI 批处理。
- **格式边界**：输入支持 `.npy`、SciPy `.npz` 和 Matrix Market `.mtx/.mm`；`tools/convert_em_matrix.py` 可将后者规范化为 `.npz`。主办方若提供专用二进制格式，仍需依据样例和字段协议增加适配器。
- **有限内存下无法保证成功**：若填充超出物理内存，只能靠重排/ILU 降级或失败退出，不会静默给出未校验结果。

## 9. 电磁仿真诊断流程（推荐用法）

```bash
# 0) 判定路径：先看相对非对称度，决定 Hermitian 还是 general
sparse-eigensolve matrix.npz --output out.npz --mode auto -k 6 --sigma <f0>

# 1) 已知目标频带：从频带中心向外走移位，避开近奇异中心
sparse-eigensolve matrix.npz --output out.npz --target-band 1.0 2.0 --adaptive-shift -k 6

# 2) 内存受限：重排 + ILU，先看峰值内存是否进入预算
sparse-eigensolve matrix.npz --output out.npz --reorder rcm \
    --inner-solver ilu --ilu-fill-factor 20 -k 6 --sigma 0.0

# 3) 精度校验：用精确 LU 结果作参考，报告子空间夹角
sparse-eigensolve matrix.npz --output out.npz --reference results/direct_reference.npz

# 4) 复现/加速实验：固定线程数
sparse-eigensolve matrix.npz --output out.npz --threads 4 -k 6 --sigma 0.0
```

诊断记录（`out.json`）中的关键字段：`problem_type`/`matrix_class`/`resolved_mode`（标准/广义问题及路径）、`mass_matrix_shape`/`mass_matrix_nnz`/`mass_matrix_dtype`（`B` 元数据）、`strategy`/`factor_kind`/`factor_nnz`（内解与填充）、`n_attempts`/`effective_shift`（移位重试与最终移位）、`arpack_iterations`/`inner_matvecs`（迭代数）、`peak_rss_mb`（峰值内存）、`subspace_angle`（若给了参考）。

## 10. 安全与合规设计

- **禁止硬编码答案**：求解器只接受真实矩阵输入，绝不含任何预设特征值/结果；所有输出由实际数值计算产生。
- **不依赖闭源服务**：全部基于 NumPy/SciPy（BSD 系许可），无网络调用（网络仅出现在可选的 `tools/fetch_matrices.py` 数据下载脚本中）。
- **可复现**：固定 BLAS 与线程数时结果逐位可复现；移位、策略、迭代数全部记录，降级运行事后可审计。
- **敏感数据**：见 [`../SECURITY.md`](../SECURITY.md)。
