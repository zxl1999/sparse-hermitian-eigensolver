# 故障场景表 / Fault-scenario report

赛题交付要求 3e 的失败场景与处置。数据由 `tools/run_fault_scenarios.py` 在**独立子进程**中实测，
带 wall-clock 超时与**父进程 RSS 内存预算**（macOS 无法设 `RLIMIT_AS`，故轮询子进程 RSS 超限即终止）。
每格含义：**精度**=相对残差 `‖Av−λv‖/((‖A‖+|λ|)‖v‖)`；**性能**=墙钟时间；**迭代次数**=ARPACK 外层迭代 / 内层 matvec；
**处置建议**见每节末。原始记录：[`fault_scenarios.json`](fault_scenarios.json)。

## near_shift — 目标位移落在特征值上 (A − σI 精确奇异)

*目标位移正好落在特征值上 (A - sigma I 精确奇异)*

| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |
|---|---|---|---|---|---|---|---|---|
| blind-retry | ✅ ok | 5.441e-17 | 0.0354s | 78.9 MB | 31 | 31 | LU (40,000) | — |
| adaptive-probes | ✅ ok | 3.670e-18 | 0.0599s | 85.7 MB | 31 | 31 | LU (40,000) | — |

**处置建议**：探测位移 `--shift-probes` + `--adaptive-shift`；不要用盲重试的小步长。

## low_freq_breakdown — 低频击穿：零附近密集谱

*零附近密集谱 (低频击穿): 8 个特征值挤在 [-1e-6, 1e-6]*

| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |
|---|---|---|---|---|---|---|---|---|
| plain | ✅ ok | 1.038e-18 | 0.0693s | 78.5 MB | 42 | 42 | LU (80,138) | — |
| target-band | ✅ ok | 3.842e-14 | 0.071s | 78.8 MB | 33 | 33 | LU (80,138) | — |

**处置建议**：已知目标频段时用 `--target-band`，候选位移从频段中心向外走，避开奇异中心。

## repeated — 重特征值 (简并谱)

*重特征值 (简并谱)*

| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |
|---|---|---|---|---|---|---|---|---|
| plain | ✅ ok | 3.601e-13 | 2.51s | 82.8 MB | 1936 | 1936 | LU (80,000) | — |
| block-ncv | ✅ ok | 7.308e-14 | 1.07s | 89.1 MB | 905 | 905 | LU (80,000) | — |

**处置建议**：加大子空间维数 `--ncv`（块 Krylov），避免 ARPACK 只收敛到简并子空间的一部分。

## clustered — 紧致谱簇

*紧致谱簇*

| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |
|---|---|---|---|---|---|---|---|---|
| plain | ✅ ok | 1.332e-18 | 0.0701s | 82.4 MB | 42 | 42 | LU (80,138) | — |

**处置建议**：收紧 `--residual-tol` 并适度加大 `--ncv`；必要时提高 `--maxiter`。

## memory_limited — 内存受限 (硬 RSS 预算)

*内存受限 (1GB 预算): 3D 7点 Laplacian A = L + I (n = 40^3 = 64000, nnz(A) ~4.4e5), 精确 LU 填充 ~1500x (~1.7GB) 超预算被终止, ILU+GMRES 降级成功 (~0.28GB)*

| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |
|---|---|---|---|---|---|---|---|---|
| direct-LU (budget) | ⛔ memory | — | — | 1.16e+03 MB / 1024 MB | — | — | — | peak RSS 1158 MB exceeded budget 1024 MB (killed) |
| ilu-gmres (budget) | ✅ ok | 5.381e-12 | 58.3s | 286 MB / 1024 MB | 309 | 3921 | ILU (5,714,777) | — |

**处置建议**：精确 LU 超预算时改用 `--inner-solver ilu`（+ `--reorder rcm`）降级；内存预算由父进程 RSS 监控强制。

## real_near_singular_Li__pli — 真实矩阵 σ=0 精确奇异 (Li__pli)

*真实电磁矩阵 Li__pli, sigma=0 精确奇异*

| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |
|---|---|---|---|---|---|---|---|---|
| adaptive-probes | ✅ ok | 4.229e-16 | 53s | 1.48e+03 MB | 32 | 32 | LU (72,816,418) | — |

**处置建议**：`--shift-probes` 给出离开奇点的绝对偏移 + `--adaptive-shift`。

## real_fill_Um__2cubes_sphere — 真实散射矩阵填充爆炸 (Um__2cubes_sphere)

*真实电磁散射矩阵 ~10 万阶, 精确 LU 填充 ~406M nnz (247x, 330s); ILU+GMRES 在 600s 内未收敛 (超时)*

| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |
|---|---|---|---|---|---|---|---|---|
| direct-RCM | ✅ ok | 1.964e-24 | 330s | 4.06e+03 MB | 30 | 30 | LU (405,858,287) | — |
| ilu-gmres | ⏱ timeout | — | — | 988 MB | — | — | — | timeout 600.0s |

**处置建议**：精确 LU 填充 ~250x；改 `--inner-solver ilu --ilu-fill-factor 20` 降到 ~19x。

## real_fill_CEMW__vfem — 真实复厄密 FEM 矩阵填充爆炸 (CEMW__vfem)

*真实复厄密 FEM 矩阵 ~9 万阶, LU 填充 ~148x, ILU 降到 ~1.56x*

| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |
|---|---|---|---|---|---|---|---|---|
| ilu-gmres | ✅ ok | 1.064e-09 | 11.3s | 449 MB | 45 | 137 | ILU (2,386,745) | — |

**处置建议**：精确 LU 填充 ~148x；改 `--inner-solver ilu` 降到 ~1.6x。

## 图例

- ✅ ok：在给定预算/超时内收敛，残差达标。
- ⏱ timeout：超过 wall-clock 超时（多为精确 LU 填充过大导致分解过慢）。
- ⛔ memory：峰值 RSS 超过内存预算，被父进程监控终止。
- ✗ error：求解器抛错（例如 ILU 丢元产生零主元 `Factor is exactly singular`）。

> 说明：`memory_limited` 的预算与超时可按机器调整（`SCENARIOS` 中的 `memory_budget_mb` / `timeout`）。
