# NOF1 / NCS1 七算法统一评测

当前主比较固定为 Single-tap LMMSE、Full-matrix LMMSE、MMSE-DFE、MMSE-PIC、UDNet、OAMP-Net2、ICR-MMSE-Net。统一入口为 `scripts.evaluate.compare`，配置为 `configs/evaluation/ber_comparison.json`。ICR 默认采用8层联合 CSI 修正与 MMSE 展开，模型与训练参数见 [ICR 说明](icr_mmse.md) 和 [算法设计](icr_mmse_algorithm_design.md)；MMSE-DFE 的公式与固定检测顺序见 [MMSE-DFE 说明](mmse_dfe.md)，OAMP-Net2 与 MMSE-PIC 的实现依据见 [对比算法说明](additional_equalizers.md)。

## 数据与指标

所有算法使用相同接收观测、已验收的 compensated CSI、既定物理记录划分和 QPSK 映射。NOF1 数据目录为 `data/processed/nof1_compact_20261026_final`，NCS1 为 `data/processed/ncs1_compact_20261026_nmse`。SNR 0→30 dB 时，输入平均 NMSE 分别为 −12.70→−18.30 dB、−6.20→−10.67 dB。

compensated CSI 按已保存的 alpha 构造 `H_init=H_ref+alpha*(H_raw-H_ref)`，属于既定的参考辅助数据构造。检测器只接收构造后的 CSI、含噪观测、已知导频及接收机信息，不单独接收参考 CSI、alpha 或数据标签。raw 视图仅用于显式诊断，检查点和缓存不得跨视图复用。

正式测试每算法、每 SNR 使用全部600帧、D1/D2和512个数据槽，共1,228,800比特。模型训练与超参数选择仅使用训练集和验证集；统一汇总再次核对数据哈希、输入视图、SNR与比特总数。

七算法结果位于 `outputs/runs/equalizer_comparison_mmse_dfe_20261031/{nof1,ncs1}/`，包含 `metrics.json`、`report.html`、七条曲线的 `ber_snr.png` 和 `nmse_snr.png`。NMSE 对每块完整矩阵的线性相对误差先平均，再转换为dB；MMSE-DFE 共用固定输入 CSI 的 NMSE，ICR另报修正后的CSI。零误码只在图中以0.5/总比特数处的空心倒三角显示，数值BER仍为零。配置更新不代表七算法已重评，原 `equalizer_comparison_unified_20261031` 和 `equalizer_comparison_compensated` 报告继续保留。

## 配置与运行

| 算法 | 当前配置 | 实现 |
|---|---|---|
| Single-tap LMMSE | 无额外参数 | `src/algorithms/MMSE/single_tap.py` |
| Full-matrix LMMSE | 完整矩阵、完整热噪声协方差 | `src/algorithms/MMSE/full_matrix.py` |
| MMSE-DFE | 完整矩阵，Cholesky，固定逆数据列顺序，硬QPSK反馈 | `src/algorithms/mmse_dfe/detector.py` |
| MMSE-PIC | 10轮，阻尼0.5，方差下限1e-10 | `src/algorithms/mmse_pic/detector.py` |
| UDNet | 22层，窗口32，隐藏维度48 | `src/algorithms/udnet/` |
| OAMP-Net2 | 10层，方差下限5e-13 | `src/algorithms/oamp_net2/` |
| ICR-MMSE-Net | 默认8层，B=16，逐层CSI修正，窗口32/16 | `src/algorithms/icr_mmse/` |

配置中的 `checkpoints` 指定每信道三个学习模型的正式检查点。ICR 已指向本次全参数展开的联合检查点：NOF1 为 `outputs/runs/icr_budgeted_unified_20261031/joint_training/nof1/icr_mmse.pt`，NCS1 为 `outputs/runs/icr_budgeted_20261031/joint_training/ncs1/icr_mmse.pt`；UDNet与OAMP继续使用各自输出目录下的 `best.pt`。开始计算前校验文件及身份；缺失或不匹配会报错。

```bash
# 默认七算法统一评测
python -m scripts.evaluate.compare --device cuda --threads 4
python -m scripts.evaluate.compare --channel NCS1 --device cuda --force mmse_pic
python -m scripts.evaluate.compare --channel NOF1 --device cuda --force mmse_dfe

# 只运行四种传统方法，不依赖学习检查点
python -m scripts.evaluate.compare --device cuda --baselines-only

# 从保存数值重新绘图
python -m scripts.evaluate.compare --plot-only
```

`--force` 只重算指定传统算法；`--force all` 同时刷新OAMP-Net2缓存。`--channel` 限定信道，`--plot-only` 不读取数据或初始化GPU，不能与 `--force` 同用。新增 MMSE-DFE 不改变其他算法的计算缓存身份。独立训练可使用较小的诊断对照，论文主比较使用上述七算法统一报告。

从头训练当前 ICR 并自动衔接完整评测时，使用独立实验目录：

```bash
python -m scripts.run.icr_experiment --output outputs/runs/icr_full_<run_id> --device cuda --threads 4
# 同一代码、配置与目录中断恢复
python -m scripts.run.icr_experiment --output outputs/runs/icr_full_<run_id> --device cuda --threads 4 --resume
```

该入口依次完成两信道共享预训练、固定与联合分支、全参数学习审计、七算法完整测试及固定分支测试。
七算法报告自动绑定本次联合权重；OAMP-Net2 和已有有效 UDNet 检查点复用，缺失的 UDNet 按正式配置训练。
运行前核对完整数据划分、compensated CSI 及基线身份。`manifest.json` 保存代码和配置，
`progress.json` 标记当前步骤、成功或失败，`logs/` 保存逐步日志；最终 `comparison/{nof1,ncs1}/`
包含两类指标图和 HTML 报告，`results_digest.json` 汇总两信道实际误码率。旧实验目录不改写。

论文正式消融采用 A–D 四组：固定 CSI + Full-matrix LMMSE、固定 CSI + MMSE Unfolding、迭代 CSI 修正 + Full-matrix LMMSE、迭代 CSI 修正 + MMSE Unfolding。它与七算法主比较分开汇总；定义、控制条件及已有结果的使用边界见 [ICR 说明第5节](icr_mmse.md#5-比较消融与输出)。单次修正、去不确定度、去门控和等预算单次修正保留为补充诊断。

## 缓存与效率

传统基线只校验并读取 `test.h5` 和 `model.h5`。完整矩阵输入保留所有接收行和数据列，按当前CSI消除已知导频。算法注册仅接受两种MMSE和MMSE-PIC；EP与Robust EP的实现及专用训练误差协方差已移除。

缓存位于 `outputs/runs/_cache/`，逐SNR、帧和D块记录完成状态、校验和及来源。计算身份绑定输入视图、算法参数、所需代码、数据内容和数值环境；更换batch、线程数、输出路径或只修改文档不使有效数值缓存失效。两种MMSE的计算身份保留。新增计算辅助函数时，在 `src/evaluation/registry.py` 和 `src/evaluation/fingerprints.py` 登记实际依赖。

损坏块重新计算，中断后从有效块续跑，文件锁阻止同键并发写入。数据在计算中变化会中止运行；报告在所有所需曲线和两类指标图完成后发布。历史报告离线重画时移除退役EP曲线。

正式效率入口 `scripts.evaluate.icr_efficiency` 默认比较六种算法，要求三个学习模型检查点，命令见 [ICR运行说明](icr_mmse.md#6-验证与运行)。每算法使用独立进程，计入CSI重建、预处理和全部迭代，禁止读取测试标签；报告初始化耗时、每块耗时、参数数和峰值内存。MMSE-PIC采用统一报告中的实际参数。

## 论文材料

论文主稿为 `outputs/paper/main.tex`，论文使用的图片直接放在主稿同目录。算法图唯一生成入口为：

```bash
python -m scripts.visualize.icr_mmse_paper_figure
```

该入口读取当前 `configs/training/icr_mmse.json`，默认直接在 `outputs/paper/` 输出PDF、SVG、PNG和 `latex_include.tex`。论文使用同目录相对文件名引用。历史稿保存在 `outputs/paper/archive/`，实验检查点和数值结果继续留在 `outputs/runs/`。
