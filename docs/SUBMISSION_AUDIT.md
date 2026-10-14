# 题目对照与提交审计

本文档只依据当前项目目录中的题目 PDF。它把“题目明确要求”“当前实现”和“不能过度宣称的边界”分开，避免把替代数据或已安装依赖误写成官方能力。

## PDF 已明确的内容

| 事项 | PDF 能确认的规范 | 当前结论 |
|---|---|---|
| 任务 | 十万阶量级非正定厄密稀疏矩阵特征值求解 | 已覆盖，最大公开样例约十万阶 |
| 输入 | NumPy、SciPy 格式矩阵 | 支持二维方阵 `.npy`、`scipy.sparse.save_npz` 生成的 `.npz`，以及通用 Matrix Market `.mtx/.mm` |
| 输出 | NumPy、SciPy 格式特征值和特征向量 | 输出合并 `.npz`；可用 `tools/export_results.py` 分离为 `.npy` |
| 故障 | 零附近密集谱、重特征值、近奇异位移、内存受限 | 已有合成/公开矩阵脚本和报告 |
| 约束 | 不得稠密化、不得硬编码、需支持降级模式 | 求解路径保持稀疏；测试 oracle 的稠密化只用于小规模校验 |
| 环境 | 多核 CPU，建议 8 核以上，16/32 GB 内存 | CPU 降级模式完整；线程控制和内存预算有记录 |
| 并行 | 提到多线程/MPI/GPU 加速 | 已集成 BLAS/OpenMP 线程、CuPy Hermitian GPU 后端和 MPI 独立任务批处理；PETSc/SLEPc 单矩阵 MPI 入口已提供，但当前 Windows 环境未完成多 rank 实测 |

## 当前 PDF 没有明确的内容

- 当前 PDF 未附带官方矩阵文件或下载地址，因此 `data/raw/` 的 SuiteSparse 矩阵只能作为公开替代基准。
- 当前 PDF 未规定输入/输出文件名、归档字段名、诊断 JSON 格式或提交目录结构。
- 当前 PDF 未规定是否提供质量矩阵 `B`；项目现已支持标准稀疏 `B` 的 `Ax=λBx`，但官方文件格式、正定性和单位映射仍需确认。
- 当前 PDF 的背景强调厄密矩阵，但工程场景提到 PML/损耗；正式提交前必须确认官方矩阵到底按 Hermitian 还是 general/complex-symmetric 解释。

## 交付时应锁定的接口

1. **输入**：确认官方矩阵是完整矩阵还是仅存一个三角；若是三角，明确使用 `--triangular hermitian` 或 `--triangular symmetric`，不要依赖猜测。
2. **矩阵类型**：先运行 `--mode auto`，记录 `matrix_class` 和 `resolved_mode`；拿到物理约定后用显式 `--mode` 复跑。
3. **目标频段**：确认频率到矩阵移位 `sigma` 的映射；项目当前只接收数值移位，不负责物理单位换算。
4. **结果**：默认提交 `.npz` 和同名 `.json`；若评测器要求独立文件，使用 `tools/export_results.py`，若要求 SciPy 稀疏容器则追加 `--scipy-vectors`，并保留字段/形状说明。
5. **广义问题**：若官方给出 `B`，使用 `--mass-matrix`；提交前确认 `B` 的格式和正定性，并复核 `A-σB` 的移位单位。项目已完成 `(A-σB)^{-1}`、ARPACK `M=B` 和原始广义残差验证。

## 并行评分风险处置

- 已提供 `--threads`、`threadpoolctl` 可选控制、线程扩展变量和 `make speedup` 基准。
- 已提供 `make capabilities`，报告 CPU 线程、MPI 独立任务批处理，以及环境中“仅安装但未接入”的 CuPy/PETSc 等依赖。
- 已用 `mpiexec -n 2` 对两个公开矩阵完成真实批处理验证；该结果证明的是任务级并行，不是单矩阵分布式分解。
- 已在 `results/SPEEDUP.md` 记录当前工作负载的实际结果：增加线程没有带来有效端到端加速，且可能增加峰值内存。
- 提交材料应写成“CPU 多线程控制 + CuPy Hermitian GPU 后端 + MPI 独立任务批处理 + 可选 SLEPc 单矩阵入口 + 完整 CPU 降级模式”。不能把 `mpi4py` 安装状态写成单矩阵分布式求解已支持。
- 如果主办方明确要求单矩阵 MPI，需在 Linux/PETSc/SLEPc 环境完成多 rank 的正确性、残差、内存和速度实测；GPU 非 Hermitian/ILU 后端仍属于后续增强。

## 提交前检查命令

```bash
python tools/check_parallel_capabilities.py
python tools/export_results.py results/eigenpairs.npz --out-dir results/exported
python -m pytest -q tests
```

最终提交包还应包含：源码、`README.md`、`PAPER.md`/Word/PDF、依赖清单、配置样例、可复现脚本、结果报告，以及公开矩阵来源和许可证说明。
