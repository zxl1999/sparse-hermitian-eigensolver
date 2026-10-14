# 降 fill-in 策略对比报告 / Fill-in Reduction Strategy Comparison

对 [中国商飞赛题](../../README.md) 的位移求逆（shift-invert）求解，**真正的内存/时间瓶颈是 `(A − σI)` 的稀疏 LU 分解填充（fill-in），而不是矩阵阶数 `n`**。本报告在 6 个真实公开矩阵上实测了三种降 fill-in 手段：

1. **矩阵重排（reordering）**：`--reorder rcm`（Reverse Cuthill–McKee）；
2. **SuperLU 排序（ordering）**：`--permc-spec {NATURAL,MMD_AT_PLUS_A,MMD_ATA,COLAMD}`；
3. **内迭代（inner solver）**：`--inner-solver ilu`，用 ILU 预条件 + GMRES 近似求解 `(A − σI)x = b`，从而完全避免精确 LU 分解。

复现命令：

```bash
python tools/benchmark_strategies.py                 # 全部矩阵，factor + solve 两步
python tools/benchmark_strategies.py --steps factor   # 只测 fill-in（更快）
```

每个 (矩阵 × 策略) 组合都在**独立子进程**中运行，并带 wall-clock 超时（本报告 `--timeout 240`，10 万阶矩阵用 `700`），因此病态填充只记为 `timeout` 而不会拖死整个套件。原始机器可读记录见 [`strategy_comparison.json`](strategy_comparison.json)。

> 说明：`factor` 步只做一次 `(A − σI)` 分解并统计 `nnz(L)+nnz(U)`（`direct` 为 SuperLU，`ilu` 为 ILU）；`solve` 步才真正跑 ARPACK 并校验残差。两步分离保证：即使某个组合的**求解**很慢，其**填充**数据仍会被记录。

## 1. 填充率（fill ratio = factor_nnz / nnz(A)）总表

| 矩阵 | n | nnz(A) | σ | COLAMD | NATURAL | MMD_AT_PLUS_A | MMD_ATA | RCM | ILU(ff=10) | ILU(ff=10)+RCM | ILU(ff=20) | ILU(ff=20)+RCM | ILU(ff=20,tol=1e-6) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Bai__mhd4800b` | 4,800 | 27,520 | 0.0 | 1.33 | 1.39 | 1.36 | 1.35 | 1.33 | **1.28** | **1.28** | — | — | — |
| `Oberwolfach__gyro` | 17,361 | 1,021,159 | 0.0 | 6.63 | 24.72 | **3.05** | 3.77 | 6.35 | 3.52 | 3.17 | — | — | — |
| `FEMLAB__waveguide3D` | 21,036 | 303,468 | 0.0 | 29.62 | ⏱ timeout | ⏱ timeout | 37.56 | **29.11** | ✗ singular | ✗ singular | 17.28 | 17.85 | — |
| `Li__pli` | 22,695 | 1,350,309 | 0.05 | 54.49 | **41.63** | ⏱ timeout | 63.94 | 54.56 | ✗ singular | ✗ singular | ✗ singular | ✗ singular | 19.01 |
| `Um__2cubes_sphere` | 101,492 | 1,647,264 | 100.0 | 253.11 | — | ⏱ timeout | — | **247.11** | ✗ singular | ✗ singular | 18.76 | 18.45 | — |
| `CEMW__vfem` | 93,476 | 1,434,636 | 0.1 | 147.57 | — | ⏱ timeout | — | 142.36 | **1.56** | 1.66 | — | — | — |

（`—` = 未测；`⏱ timeout` = 超过该矩阵的单步超时；`✗ singular` = ILU 因丢元产生零主元而失败。数字越低越好，加粗为该矩阵最优。）

对应分解耗时（秒，同一台 10 核 / 16 GB 机器）：

| 矩阵 | COLAMD | NATURAL | MMD_AT_PLUS_A | MMD_ATA | RCM | ILU(ff=10) | ILU(ff=20) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Bai__mhd4800b` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| `Oberwolfach__gyro` | 0.32 | 4.26 | 0.11 | 0.26 | 0.31 | 0.29 | — |
| `FEMLAB__waveguide3D` | 4.27 | >240 | >240 | 6.67 | 4.16 | ✗ | 17.74 |
| `Li__pli` | 24.79 | 19.24 | >240 | 31.70 | 25.29 | ✗ | ✗ |
| `Um__2cubes_sphere` | 381.17 | — | >700 | — | 334.77 | ✗ | 33.15 |
| `CEMW__vfem` | 499.51 | — | >700 | — | 511.54 | 5.97 | — |

## 2. 关键发现

1. **降 fill-in 收益可达 1~2 个数量级，且是"必做项"而非"可选项"。**
   在 10 万阶矩阵上，精确 LU 直接爆掉：`Um__2cubes_sphere` COLAMD **253×** 填充（4.17 亿非零元、381 s），`CEMW__vfem` **148×**（2.12 亿非零元、500 s）。而改用 ILU 内迭代后，`CEMW__vfem` 仅 **1.56×**（约 224 万非零元、6 s）——**内存与时间都降了约 100 倍**。

2. **没有"万能最优"排序，必须按矩阵选。** 四种排序互有胜负：
   - `Oberwolfach__gyro`：`MMD_AT_PLUS_A` 最优（3.05×，比 COLAMD 6.63× 省一半）；
   - `Li__pli`：`NATURAL` 反而最优（41.63× < COLAMD 54.49×）；
   - `FEMLAB__waveguide3D`：`RCM` 与 `COLAMD` 接近（29.1× / 29.6×），而 `MMD_AT_PLUS_A` **灾难性**（超时）；
   - 两个 10 万阶矩阵：`RCM` 略优于 `COLAMD`（247× vs 253×，142× vs 148×）。
   → 结论：**排序选择本身是矩阵相关的，建议在求解前做一次便宜的"排序试算"（本工具即为此设计）**。

3. **`NATURAL` 可能灾难性，绝不能作为默认。** `gyro` 上 `NATURAL` 的填充是 `MMD_AT_PLUS_A` 的 8 倍；`waveguide3D`（复 3D curl-curl）上直接超时。SciPy 默认 `COLAMD` 是安全基线。

4. **ILU + GMRES 是最强的降内存武器，但需要调 `fill_factor`（必要时再调 `drop_tol`）。**
   - 默认 `fill_factor=10` 在 3 个较难矩阵上抛 `Factor is exactly singular`——这**不是** `A − σI` 奇异，而是 ILU 丢元（`drop_tol`）产生的零主元。把 `fill_factor` 提到 20 即可修复（`waveguide3D` 17.28×、`2cubes_sphere` 18.76×）。
   - `Li__pli` 更顽固：`fill_factor=20` 仍失败，需同时把 `--ilu-drop-tol` 收到 `1e-6`（→ **19.01×**，相比精确 LU 的 54.49× 省近 3 倍）。也就是说 **ILU 参数应按矩阵调**，这也是后续做"自适应 ILU"的动机。
   - 在 `gyro` 上 ILU（3.52×）已接近最优精确排序（3.05×），但内迭代引入的 GMRES 收敛成本使整体不一定更快。

5. **精确 LU 的填充随问题规模急剧恶化。** 从 `gyro`（n=17k，6.6×）到 `Li__pli`（n=23k，54×）到 10 万阶（148×~253×），填充率的增长远快于 `n`。这正是赛题"低内存"要求的核心矛盾。

## 3. 特殊点：σ 处的奇异性

- `Li__pli` 在 σ=0.0 时，**所有**排序都报 `Factor is exactly singular`——因为 0 位于（或极接近）其谱中。改用 σ=0.05（略偏离）后所有排序均可分解。这验证了求解器的**自适应位移**逻辑（失败后按 `shift_step` 偏移重试）是必要的。
- `Bai__mhd4800b` 在 σ=1.0、`Li__pli` 在 σ=0.0 均为精确奇异；求解器默认从 `σ` 出发、失败后自动 ±步长重试，已在测试中覆盖。

## 4. 建议

1. **默认组合**：`--permc-spec MMD_AT_PLUS_A` 或 `COLAMD`（二者在不同矩阵上互补），先试算再定；避免 `NATURAL`。
2. **10 万阶以上**：优先 `--inner-solver ilu --ilu-fill-factor 20`，把内存从"亿级非零元"压到"千万级"；仅当 GMRES 收敛太慢时才退回精确 LU。
3. **ILU 失败时**：先提高 `--ilu-fill-factor`（10→20→50）；若仍失败，再把 `--ilu-drop-tol` 调小（`1e-4→1e-6`）。实测 `Li__pli` 正是靠 `fill_factor=20 + drop_tol=1e-6` 才成功（19.01×）。更强的预条件（ILUT / 多重网格）留作后续工作。
4. **务必先用本工具在目标矩阵上试算排序**，再决定提交时的默认配置——排序收益是矩阵相关的。

## 5. 与主报告的衔接

本报告是 [PATH_COMPARISON.md](PATH_COMPARISON.md) §4 第 5 条的实测支撑。两点与主报告一致：

- **阻塞项不变**：官方矩阵是**复厄密**还是**复对称（有损/PML）**仍需与主办方确认——它决定走 `eigsh` 还是 `eigs`，进而影响 fill-in 曲线（本报告 `waveguide3D` 与 `CEMW__vfem` 均为 complex128，即真实现场常见的复矩阵）。
- **瓶颈结论不变**：位移求逆的 LU 填充（而非 `n`）才是内存/时间瓶颈；本报告给出的三种手段可把该瓶颈降低 1~2 个数量级。
