# ICR-MMSE-Net：当前统一算法与运行方式

当前设计为 **8层残差 MMSE Unfolding + 共享带状 CSI 修正器**。NOF1、NCS1 都学习逐层物理参数和局部神经校准网络，分别训练权重。公式、参数和冻结测试见 [算法设计](icr_mmse_algorithm_design.md)。

## 1. 输入与模型

统一使用已验收的 compensated CSI 和既定划分。NOF1 输入平均 NMSE 为 −12.70→−18.30 dB，NCS1 为 −6.20→−10.67 dB，SNR为0→30 dB。算法输入包含接收观测、完整初始CSI、公开导频、物理位置/掩码和噪声参数；真实符号与参考CSI仅供监督及预测后的指标计算。

默认 `complex64`、对角噪声加权、修正半带宽16、局部窗口32、中心16、隐藏宽度48。修正器采用 `14(2B+1)+5` 个输入通道，修正仅作用于带内元素，所有带外耦合继续参与检测。

每层由 LMMSE 正规方程导出预条件残差方向，加入非负可学习 CSI 加载，并以精确二次方向线搜索和 `(0,2)` 内的可学习系数控制更新。QPSK距离 logits 与局部网络校准相加，概率阻尼使用稳定的 `log_softmax/logaddexp`。每层学习步长系数、加载、温度、不确定度和软阻尼，以及局部 MLP；共197,544个检测器参数，其中40个物理参数。两信道都执行完整神经分支。

## 2. 当前配置和训练

唯一训练配置为 `configs/training/icr_mmse.json`。共同结构放在 `model`，共同训练设置放在 `training`，`channel_overrides` 记录信道独立的加载初值、学习率和监督权重；禁止仅为某个信道冻结神经网络。旧标量配置及分散的 `icr_budgeted_*.json` 已合并清理。

训练入口依次执行共同预训练、固定CSI分支和联合分支。两分支共享同一个阶段A检查点：

| 分支 | 阶段/轮数 | 学习内容 |
|---|---|---|
| `warmup` | A：10 epoch | 固定CSI，全部物理参数和神经网络 |
| `fixed` | 复用A，继续10 epoch | 固定CSI，全部检测器参数 |
| `joint` | 复用A，B：10、C：10 epoch | B仅CSI修正器；C学习检测器与修正器 |

检测损失为比特交叉熵，末层权重0.9、其余7层各0.1/7。阶段C的完整矩阵CSI NMSE使用相同层权重，再乘以 `lambda_h`；阶段B仅对所有层和样本的CSI NMSE均匀平均。阶段A按验证BER选型，阶段B优先验证NMSE，阶段C优先考虑CSI是否改善，再比较验证BER和NMSE。训练种子20261028，batch 8，SNR交错，`dense_device`，预取2批，8192 MiB有界主存缓存。具体分组学习率见算法设计第3.7节。

```bash
# 服务器：依次训练两信道及两分支
python -m scripts.run.icr_mmse --device cuda --threads 4
# 只训练一个信道，或恢复同一配置
python -m scripts.run.icr_mmse --channel NOF1 --device cuda --resume
# 单独执行某个阶段，fixed/joint要求先有同根目录下的warmup
python -m scripts.train.icr_mmse --channel NCS1 --stage warmup --device cuda
python -m scripts.train.icr_mmse --channel NCS1 --stage joint --device cuda
```

新训练目录为 `outputs/runs/icr_mmse_current/{warmup,fixed,joint}/<channel>/`。`--output` 指定新的实验根目录，所有分支及共享预训练路径共同切换。续训校验实际训练、数据和源码身份；不能用整理后的新配置直接续接历史身份不同的目录。

本地仅运行CPU小样本验证；完整训练仍在服务器进行。性能测量入口为 `scripts.evaluate.icr_training_perf`，它会解析对应信道的实际模型设置，计时和缓存参数由自己的CLI控制。

## 3. 已验收检查点和预算复核

当前正式推理权重如下，六算法配置已指向这两个联合模型：

| 信道 | 固定CSI | 完整模型 |
|---|---|---|
| NOF1 | `icr_budgeted_unified_20261031/fixed_final/nof1/icr_mmse.pt` | `icr_budgeted_unified_20261031/joint_training/nof1/icr_mmse.pt` |
| NCS1 | `icr_budgeted_20261031/fixed_final/ncs1/icr_mmse.pt` | `icr_budgeted_20261031/joint_training/ncs1/icr_mmse.pt` |

路径均相对 `outputs/runs/`。旧文件位置保持不变，历史训练记录与权重摘要一同保留。推理通过 `prepare_frozen_inference` 缓存固定参数变换和窗口几何，数值等价检查覆盖真实验证输入；不修改推理公式或保存新的训练权重。

```bash
# 读取既有四组冻结报告，生成BER/NMSE与逐层图
python -m scripts.evaluate.icr_budgeted_report
# 核验并复用同身份的已完成预算评测
python -m scripts.run.icr_budgeted_study --split validation --device cuda --resume
python -m scripts.run.icr_budgeted_study --split test --device cuda --resume
```

四组平均BER均优于历史和同修正器权重LMMSE，实测总耗时不超过原Unfolding预算。全部展开层及CSI修正计入稳态耗时，一次性加载/准备单列。历史NOF1为22层、新版8层，NCS1均8层，因此这批预算修复结果不替代相同深度的正式A–D消融。

NOF1固定CSI相对未训练初始化没有显著学习收益，部分0 dB点仍差于LMMSE，联合CSI NMSE也未全面改善。单种子、物理记录配对区间不能证明训练随机性稳定或神经校准的独立贡献。完整负面结果和区间保存在冻结报告中。

## 4. 六算法主比较

比较 Single-tap MMSE、Full-matrix LMMSE、MMSE-PIC、UDNet、OAMP-Net2 和当前 ICR-MMSE-Net，配置为 `configs/evaluation/ber_comparison.json`。新汇总写到 `outputs/runs/equalizer_comparison_unified_20261031/`，运行方法见 [统一评测](equalizer_comparison.md)。

```bash
python -m scripts.evaluate.compare --device cuda --threads 4
```

更新检查点配置不会使旧报告自动变成新报告；需要正式重评后才能引用新六算法表。此次配置与仓库同步不自动重新启动大规模计算。

## 5. 比较、消融与输出

论文核心消融采用下列四组，与训练阶段A/B/C区分：

| 版本 | CSI处理 | 均衡器 |
|---|---|---|
| A | 固定CSI | Full-matrix LMMSE |
| B | 固定CSI | 当前MMSE Unfolding |
| C | 迭代CSI修正 | Full-matrix LMMSE |
| D | 迭代CSI修正 | 当前MMSE Unfolding |

B/A检查固定CSI下的检测器替换，C/A检查传统检测器下的修正收益，D/B检查展开检测器下的修正收益，D/C检查已有修正时的检测器作用。C保留多轮CSI修正与软反馈，A只做一次有效求解。B/D使用同一展开深度及共同预训练，C/D修正结构及轮数一致。参数和检查点只由训练/验证确定，不预设每个SNR均改善。

旧 `icr_ad_ablation_20261030` 检测器初始化审计显示最终B/D未实际学习，结果仍保留，但不能将其差异归因于参数学习。单次修正、去不确定度、去门控及等预算一次修正属于补充诊断；已有产物保留，旧自动搜索/细项训练流水线已移除。

正式A–D自动运行入口如下。`--trained-root` 指向当前配置完成的共同预训练及固定/联合分支，入口核验训练源码、数据、配置和逐层参数学习后复用B/D；A没有待学习检测器，C独立训练具有LMMSE软反馈的CSI修正器，沿用D的8轮修正及B/C阶段预算。所有模型先冻结，再运行验证集和测试集，测试结果不参与选型，也不要求预设的性能排序。

```bash
python -m scripts.run.icr_ablation --trained-root outputs/runs/icr_full_20261031_095549 --output outputs/runs/icr_ad_current_<run_id> --device cuda --threads 4
# 中断后按同一代码、配置、数据和权重恢复
python -m scripts.run.icr_ablation --trained-root outputs/runs/icr_full_20261031_095549 --output outputs/runs/icr_ad_current_<run_id> --device cuda --threads 4 --resume
```

输出包含 `progress.json`、冻结权重清单、检测器学习审计、每信道/划分/组别的逐层指标、A–D汇总JSON/CSV、BER/NMSE对比图、四项物理记录配对BER差异区间及可传回本地的 `reports.zip`。失败时进度记录异常，既有检查点和历史实验均保留。

每次完整评测至少生成BER–SNR、平均完整矩阵NMSE–SNR，保留实际误码数、逐层结果、数据/代码/权重摘要、计时及物理记录配对统计。预算复核保留的 `legacy_20261030.py` 与 `legacy_lmmse_20261030.py` 是原推理快照，不能删除或当成活动训练模型。

## 6. 验证与运行

本地CPU验证覆盖新配置解析、两分支共享预训练、分组学习率、逐层参数学习审计及推理等价性。服务器按 `main` 同步代码后读取原数据和检查点。Git只保存代码、配置、测试与文档；数据、训练权重、旧结果和论文产物留在各自输出目录。
