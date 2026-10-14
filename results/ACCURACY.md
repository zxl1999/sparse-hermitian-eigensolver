# 精度报告 (Accuracy report)

本报告用两种方式量化求解精度：小规模问题对照稠密精确解（NumPy `eigh`/`eig`），
大规模问题用两种独立内解策略（精确 LU 与 ILU+GMRES）互校。
子空间夹角由 `diagnostics.subspace_angle` 给出（SVD 主角度），特征值误差按最近值匹配。

## 1. 稠密精确解对照 (dense oracle, small n)

| 算例 | n | 模式 | 最大残差 | 子空间夹角(最大/平均) | 特征值误差 | 正交性误差 | Arnoldi 迭代 |
|---|---|---|---|---|---|---|---|
| indefinite (Hermitian) | 400 | hermitian | 9.77e-14 | 1.21e-06° / 2.01e-07° | 2.26e-17 | 3.66e-15 | 33 |
| near_shift (Hermitian) | 400 | hermitian | 2.03e-15 | 8.54e-07° / 2.85e-07° | 4.34e-18 | 1.63e-15 | 31 |
| complex_hermitian | 400 | hermitian | 2.06e-15 | 1.21e-06° / 4.86e-07° | 2.26e-17 | 2e-15 | 31 |
| complex_symmetric (general) | 300 | general | 1.82e-16 | 1.21e-06° / 2.01e-07° | 4.72e-17 | 1.16 | 41 |

## 2. 求解器互校 (solver vs. solver, large n)

| 算例 | n | 模式 | 精确 LU 残差 | ILU 残差 | 子空间夹角(最大) | 特征值误差 | LU 填充比 | ILU 填充比 |
|---|---|---|---|---|---|---|---|---|
| indefinite n=20000 | 20000 | hermitian | 1.18e-16 | 9.18e-15 | 2.26e-06° | 1.75e-16 | 1.34x | 1.32x |
| near_shift n=20000 | 20000 | hermitian | 7.82e-17 | 3.41e-16 | 1.21e-06° | 1.63e-18 | 2x | 2x |
| Bai__mhd4800b (real) | 4800 | hermitian | 6.27e-24 | 1.57e-23 | 1.48e-06° | 8.08e-27 | 1.3x | 1.28x |

## 结论

- 小规模对照稠密精确解：子空间夹角与特征值误差均在求解容差量级（见上表），说明 shift-invert 路径正确。
- 大规模 LU 与 ILU 互校：两者子空间夹角极小，说明 ILU+GMRES 的低内存替代在精度上可接受。
- 残差按 `||A v - λ v|| / ((||A|| + |λ|) ||v||)` 归一化，并始终针对原始矩阵校验。
