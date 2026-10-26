# EPNet

基于 WATERMARK NCS1／NOF1 的水声通信 AI 信道均衡研究项目，统一使用 Python。
通过真实接收机链路生成数据和不完美 CSI，当前使用 MMSE 与 EP 进行性能对比。

## 文件目录结构

```text
EPNet/
├── configs/data/             # NCS1／NOF1 固定数据生成配置
├── data_generation/
│   ├── scripts/              # 数据生成与物理链路检查入口
│   ├── src/                  # 发射、信道、接收机、CSI 估计及导出
│   └── tests/                # 数据生成测试
├── src/
│   ├── algorithms/           # MMSE、EP 均衡算法
│   ├── common/               # 数值后端与实验元数据
│   ├── data/                 # 数据读取与预处理
│   └── evaluation/           # 指标计算、图表与报告
├── scripts/evaluate/         # 均衡评测入口
├── tests/                    # 算法、数据和评测测试
├── docs/                     # 设计与运行说明
├── data/
│   ├── raw/watermark/        # 原始 WATERMARK 源包
│   ├── interim/              # 可再生成的校准缓存
│   └── processed/            # 接收机数据与估计 CSI
├── outputs/runs/             # 按实验保存的结果
├── AGENTS.md                 # 项目开发约定
└── pyproject.toml            # Python 依赖与测试配置
```
