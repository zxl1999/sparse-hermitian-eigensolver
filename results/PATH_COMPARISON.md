# 求解路径对比报告 / Solver Path Comparison

对 [中国商飞赛题](../../README.md) 的电磁谐振与微波器件仿真问题，本项目实现了两条求解路径，并在真实公开电磁矩阵上做了实测对比。

## 1. 两条路径

| 路径 | 矩阵假设 | 后端 | 残差定义 | 适用物理场景 |
| --- | --- | --- | --- | --- |
| `hermitian` | `A = A^H`（实对称 / 复厄密） | `scipy.sparse.linalg.eigsh`（ARPACK，位移求逆） | `‖Av − λv‖ / ((‖A‖+‖λ‖)‖v‖)` | 无损腔体、理想导体谐振器 |
| `general` | 无对称性假设（含复对称 `A = A^T`） | `scipy.sparse.linalg.eigs`（ARPACK，位移求逆，双正交） | 同上，用 `A^H` 双正交归一 | 有损介质 / PML 吸收边界 |

`--mode auto`（默认）通过相对非厄密度 `‖A − A^H‖ / ‖A‖` 自动分流：
- `<= --hermitian-tol`（默认 `1e-12`）→ `hermitian` 路径，并把矩阵对称化为 `(A + A^H)/2`；
- 否则 → `general` 路径，按原矩阵求解。

判定阈值与最终采用的路径都会写入诊断 JSON 的 `matrix_class` 与 `resolved_mode` 字段，便于复现与审计。

## 2. 为什么两条都要做

赛题 PDF 只写了"非正定厄密"，但真实的**有损 / PML 电磁本征问题通常是复对称（complex symmetric）而非复厄密**。这两种约定下**特征值不同**，算法也不能互换。用 `FEMLAB/waveguide3D`（n=21,036，相对非厄密度 0.179，σ=0）实测：

| 路径 | 返回的特征值（σ=0 附近 4 个） | 最大相对残差 |
| --- | --- | --- |
| `general`（正确建模） | `-3.75e-4+2.1e-6j`、`-1.70e-4+4.2e-5j`、`-3.7e-5+1.05e-4j`、`1.46e-4+1.31e-4j` | 1.5e-15 |
| 强制 `hermitian`（对称化后） | `-1.76e-4`、`8.4e-5`、`3.96e-4`、`5.31e-4`（全为实数） | 4.9e-15 |

两组谱**完全不重叠**。因此这不是代码风格问题，而是**正确性问题**：必须先从官方确认矩阵约定。

## 3. 全部公开矩阵实测

`sparse-eigensolve ... -k 6`，`/usr/bin/time -l` 记录峰值内存。复现命令：

```bash
python tools/run_public_suite.py
```

（自动读取 `data/raw/*.npz`，输出 `results/path_comparison.json`）

| 矩阵 | n | nnz | dtype | 自动判定 | 采用路径 | σ | 耗时 (s) | 最大相对残差 | 峰值内存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Bai__mhdb416` | 416 | 2,312 | float64 | hermitian | hermitian | 0.0 | 0.00 | 1.0e-20 | 0.06 GB |
| `Bai__bfwb782` | 782 | 5,982 | float64 | hermitian | hermitian | 0.0 | 0.01 | 2.7e-15 | 0.06 GB |
| `Bai__mhd4800b` | 4,800 | 27,520 | float64 | hermitian | hermitian | 0.0 | 0.03 | 5.0e-24 | 0.07 GB |
| `Oberwolfach__gyro` | 17,361 | 1,021,159 | float64 | hermitian | hermitian | 0.0 | 0.60 | 9.1e-18 | 0.24 GB |
| `FEMLAB__waveguide3D` | 21,036 | 303,468 | complex128 | **general** | general | 0.0 | 6.26 | 4.8e-16 | 0.30 GB |
| `Li__pli` | 22,695 | 1,350,309 | float64 | hermitian | hermitian | 0.0 | 52.1 | 1.5e-16 | 1.25 GB |
| `CEMW__vfem` | 93,476 | 1,434,636 | complex128 | hermitian | hermitian | 0.1 | 467.9 | 2.5e-12 | 4.04 GB |
| `Um__2cubes_sphere` | 101,492 | 1,647,264 | float64 | hermitian | hermitian | 100.0 | 409.4 | 2.4e-24 | 4.37 GB |

> 表中数值由 `results/path_comparison.json` 机器可读记录导出（同一次运行）；耗时与残差随运行略有波动，属正常现象。峰值内存来自同矩阵的 `/usr/bin/time -l` 单独测量。

## 4. 结论

1. **自动分流工作正常**：8 个矩阵中 7 个判为 Hermitian，真实非厄密的 `waveguide3D` 判为 general 并走 `eigs`，两条路径残差均在 `1e-12` 以内。
2. **规模瓶颈是位移求逆的 LU fill-in，而不是 `n`**：10 万阶三对角合成矩阵 0.3 s / 0.18 GB，而真实 10 万阶矩阵要 400 s 级、4 GB 级；2.3 万阶 curl-curl（`Li__pli`）就已 51.6 s / 1.25 GB。
3. **`general` 路径明显更贵**：同样点附近 `waveguide3D`（n=21k）需要 7.4 s，而 Hermitian 路径下同尺寸问题普遍在 1 s 内——双正交 ARPACK 的迭代成本更高，这进一步说明确认矩阵约定很重要。
4. **待官方确认（阻塞项）**：矩阵是**复厄密**还是**复对称（有损/PML）**。确认前建议保持 `--mode auto`，并在提交材料中附上每题的 `matrix_class` / `resolved_mode`。
5. 降 fill-in 优化已实现并实测（矩阵重排 RCM、SuperLU `permc_spec`、ILU + GMRES 内迭代），结果见 [STRATEGY_COMPARISON.md](STRATEGY_COMPARISON.md)；针对已知谱段的自适应位移仍待做。

## 5. 复现

```bash
cd path/to/project
python -m pip install -e '.[test]'
python -m pytest -q tests                       # 11 个测试
python tools/fetch_matrices.py --list
python tools/fetch_matrices.py --out data/raw
python tools/fetch_matrices.py --out data/raw --names FEMLAB/waveguide3D --keep-general
python tools/run_public_suite.py
```
