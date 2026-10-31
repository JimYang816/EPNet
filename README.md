# EPNet

基于 WATERMARK NCS1／NOF1 的水声通信 AI 信道均衡研究项目，统一使用 Python。
基于已验收的 compensated CSI，对比 Single-tap MMSE、Full-matrix LMMSE、MMSE-PIC、
UDNet、OAMP-Net2 和 ICR-MMSE-Net；当前 ICR 默认采用 8 层联合 CSI 修正与 MMSE 展开。

## 文件目录结构

```text
EPNet/
├── configs/
│   ├── data/                 # NCS1／NOF1 固定数据生成配置
│   ├── training/             # ICR-MMSE-Net、UDNet、OAMP-Net2 训练配置
│   └── evaluation/           # 六算法统一评测配置
├── data_generation/
│   ├── scripts/              # 数据生成与物理链路检查入口
│   ├── src/                  # 发射、信道、接收机、CSI 估计及导出
│   └── tests/                # 数据生成测试
├── src/
│   ├── algorithms/           # 六种均衡算法实现
│   ├── common/               # 数值后端与实验元数据
│   ├── data/                 # 数据读取与预处理
│   ├── training/             # 学习型算法训练与验证选型
│   └── evaluation/           # 指标计算、六算法对比与效率评测
├── scripts/
│   ├── train/                # 独立训练入口
│   ├── run/                  # 训练、选型及消融流水线
│   ├── evaluate/             # 统一评测与必要诊断入口
│   ├── infer/                # 推理入口
│   └── visualize/            # 论文算法图生成入口
├── tests/                    # 算法、数据、训练与评测测试
├── docs/                     # 设计与运行说明
├── data/
│   ├── raw/watermark/        # 原始 WATERMARK 源包
│   ├── interim/              # 可再生成的校准缓存
│   └── processed/            # 接收机数据与估计 CSI
├── outputs/
│   ├── runs/                 # 正式实验、消融及保留的历史结果
│   └── paper/                # 论文主稿 main.tex、图片及历史稿 archive/
├── AGENTS.md                 # 项目开发约定
└── pyproject.toml            # Python 依赖与测试配置
```
