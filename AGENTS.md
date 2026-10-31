# 项目说明

本项目用于水声通信 AI 信道均衡算法研究，基于 WATERMARK 信道构建真实接收机链路数据集，使用接收机估计得到的不完美 CSI 对 MMSE 基线与学习型均衡算法进行统一指标对比。

## 文档与当前实验范围

- 数据生成已完成，后续所有均衡算法统一使用当前已验收的数据集及既定划分进行训练和测试。
- 当前 ICR 算法使用统一8层残差 MMSE Unfolding：NOF1、NCS1 均学习逐层物理参数和局部神经校准网络，禁止按信道冻结神经分支。唯一训练配置为 `configs/training/icr_mmse.json`，允许各信道独立权重及验证确定的学习率/加载初值。当前设计见 `docs/icr_mmse_algorithm_design.md`。
- 主对比包含 Single-tap LMMSE、Full-matrix LMMSE、MMSE-PIC、UDNet、OAMP-Net2 和 ICR-MMSE-Net；EP、Robust EP 不再参与性能或效率对比。
- 论文正式消融采用 A–D 四组：A＝固定 CSI + Full-matrix LMMSE，B＝固定 CSI + MMSE Unfolding，C＝迭代 CSI 修正 + Full-matrix LMMSE，D＝迭代 CSI 修正 + MMSE Unfolding（完整模型）。单次修正、去不确定度、去门控及等预算单次修正保留为补充诊断，不列入正文核心消融；定义与结果使用边界见 `docs/icr_mmse.md` 第5节。
- 默认训练、验证、测试必须使用已验收的 **compensated CSI**：NOF1 平均 NMSE 随 SNR 0→30 dB 为 **−12.70→−18.30 dB**，NCS1 为 **−6.20→−10.67 dB**。未经用户明确要求，不得改为 raw CSI 或修改这些目标。raw 数据保留，仅供显式诊断；不得复用 raw 视图的训练检查点、特征或评测缓存。
- `README.md` 仅保留项目简要说明及文件目录结构；未经用户明确允许，不得修改。
- NCS1 与 NOF1 使用 `configs/data/` 中的两份固定配置，共用同一实际 DPSS 时间基与延迟支撑。
- 原始 WATERMARK 源包放在 `data/raw/watermark/`；生成数据、缓存和实验结果不得提交到 Git。

## 实现约定

- 按功能将代码及相关文件放入对应目录，命名清晰恰当，不随意散放文件。
- 整个项目统一使用 **Python** 实现，包括数据生成、信道均衡算法、学习型算法训练、评测指标计算与性能对比。
- 数据生成部分独立放在根目录 `data_generation/`，保留 `scripts/`、`src/` 和 `tests/` 布局。数据生成代码放在 `data_generation/src/`，包括发射信号生成、WATERMARK 信道仿真、接收机处理、CSI 估计及数据整理与导出；运行入口和测试分别放在 `data_generation/scripts/` 与 `data_generation/tests/`。
- 数据读取与预处理代码放在 `src/data/`，算法、训练及评测代码分别放在 `src/algorithms/`、`src/training/` 和 `src/evaluation/`；对应运行入口放在根目录 `scripts/`。
- `data/processed/` 保存真实接收机链路数据及估计得到的不完美 CSI，供训练与评测使用。
- 实验结果存放在 `outputs/runs/`；旧实验与消融结果必须保留，清理仅针对确认无用的临时文件和测试缓存。
- 论文主稿与相关 TeX 材料统一存放在 `outputs/paper/`；论文中使用的所有图片必须直接放在 `main.tex` 同目录下，不放入图片子目录。
- 每次实验结果指标图至少包括BER-SNR、平均NMSE-SNR。

## 开发与运行方式

- **本地**：在 `D:\SCI\code\EPNet` 生成和修改代码，仅用 CPU 和少量测试数据验证能跑通即可。
- **Git 提交**：提交信息统一使用中文。
- **代码同步**：本地验证通过后推送至 GitHub 仓库 `https://github.com/JimYang816/EPNet.git`，服务器拉取对应版本后再正式运行。
- **远程运行**：通过 `ssh lab448` 连接 Linux 服务器，项目目录为 `/home/fay/JimYang/EPNet`。运行前检查可用资源，根据实际配置使用 GPU 进行正式计算。
- **数据与结果**：完整数据留在服务器，实验结果通过 SSH 文件传输带回本地；GitHub 只同步代码、测试、配置和文档。
