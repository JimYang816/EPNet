# SNR 交错训练与门控配对实验

本实验保持 ICR-MMSE-Net 的门控、修正网络、阻尼和均衡器结构不变，只改变训练样本顺序和验证集模型选择规则。使用当前验收的 compensated CSI 及固定 train/validation/test 划分，不重新生成数据，不复用 raw CSI 产物。

## 固定对照条件

入口为 `python -m scripts.run.icr_gate_controls`，配置为 `configs/training/icr_gate_interleaved.json`，输出为 `outputs/runs/icr_gate_interleaved_20261029/`。

- NOF1 固定 22 层、`lambda_h=0.01`；NCS1 固定 8 层、`lambda_h=1`。这些值沿用原验证集选型，本轮不重新搜索。
- 每个信道使用一个配对种子：20261028。两分支共用其阶段 A 预热权重，完整网络前向仅 `use_gate` 不同。当前运行用于快速比较，不据此声称跨种子稳定性。
- 阶段 A/B/C 各 10 epoch，batch 8；A/B 学习率 1e-3，C 均衡器学习率 1e-4，C 修正器和阻尼学习率 1e-5。修正次数、不确定度反馈和噪声模式不变。
- 训练使用 `dense_device`、预取 2 批、12288 MiB 主存参考缓存；评测使用既有完整矩阵 `dense` 路径，batch 2。
- 正式运行开启 PyTorch 确定性算法，禁止 cuDNN benchmark，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；如遇不支持的确定性算子则明确失败，不静默回退。

## 交错采样

每个 batch 内保持相同 SNR 和 D 块，以保留原 batch 大小、实际 DPSS 时间基和补偿系数。每一轮依次覆盖所有 SNR，轮内 SNR 顺序随机；物理帧与 D 块顺序按 epoch 打乱。每个 `(SNR, frame, D-block)` 每轮 epoch 恰好出现一次，保留尾批，不增加或丢弃样本。

顺序由 `SeedSequence([训练种子, 阶段编号, epoch])` 唯一决定，与模型随机数消耗、门控开关、预取线程及先前运行无关。检查点保存阶段和 epoch，训练历史记录每轮 `sampling_seed`；恢复时重新构建相同顺序。采样方案、控制参数、源码和确定性设置进入续训身份校验。验证与测试仍按原固定顺序遍历。

## 逐 SNR 验收

每个 SNR 分别累计 BER 和完整参考矩阵 NMSE。NMSE 先按每个 D 块完整参考能量归一化，再平均线性值，图中最后转 dB。

- 阶段 A 保持原验证 BER 选择规则。
- 阶段 B 以进入 B 时的阶段 A 权重为固定参考；阶段 C 以进入 C 时选中的 B 权重为固定参考。
- 每个 SNR 的 BER 和最终层完整矩阵 NMSE 均不得超过相应参考值，默认所有容差为零。仅在全部 SNR 合格的候选中，B 优先选择平均 NMSE，C 按既有 CSI 改善与平均 BER 规则选择。
- 阶段起始权重始终保留为候选；没有合格改进时回退。参考界限不随 epoch 逐步放宽，每轮记录违反界限的 SNR、指标、数值和界限。
- 两组使用相同规则，但 C 阶段的参考数值来自各自选中的 B，不要求两个不同模型的 B 指标相同。

零容差是本轮预先固定的保守验收条件。若所有训练候选都被拒绝，应报告回退及拒绝原因，不能看过测试结果后放宽容差。验证约束不保证测试集逐 SNR 无退化。

## 执行与产物

```bash
.venv/bin/python -m scripts.run.icr_gate_controls --device cuda --gpu 0 --threads 4
# 中断后使用相同代码、配置和设备
.venv/bin/python -m scripts.run.icr_gate_controls --device cuda --gpu 0 --threads 4 --resume
```

监督入口取得单 GPU 文件锁并串行执行两个信道/种子组合。每对两模型均完成验证选择后才读取 test。启动、当前阶段、完成或失败信息保存在 `_control/status.json`，Git 提交与配置保存在 `_control/manifest.json`。

每对保留 `gate/`、`no_gate/` 的 A/B/最终检查点、训练记录、逐层评测及 `pair.json`。成功完成一对后删除冗余的 `latest.pt` 恢复副本，保留正式检查点及审计信息。汇总生成 `comparison.json`、`comparison.csv`、`seed_summary.json`、`ber_snr.png/svg` 与 `nmse_snr.png/svg`。

BER 差定义为 `gate - no_gate`，负值表示门控改善。每个种子内用相同物理记录做配对 bootstrap，记录不是独立比特；跨种子均值和标准差单独报告，不把种子当作新的独立物理记录。汇总曲线对种子求均值，NMSE 在转 dB 前平均线性值。部分完成的图明确标出已经完成的配对种子数，不代表全部实验完成。

## 有限预算的末层均衡器诊断

入口为 `python -m scripts.run.icr_final_detector`，配置为 `configs/training/icr_final_detector.json`。从上述实验同一种子的两份 B 检查点分别继续，只运行 NOF1，每组最多两轮 C，不重新执行 A/B，不读取 test。输出放在独立的 `outputs/runs/icr_final_detector_20261029/`，原实验全部保留。

`stage_c_final_equalizer_only=True` 冻结修正器、阻尼和最后一层之前的全部均衡器。最后一次 CSI 修正在最后一层均衡之前完成，因此末层参数更新不会改变任何后续 CSI 或反馈路径；仅冻结修正器而更新所有均衡器不具备这一性质。训练损失为末层交叉熵加末层 CSI NMSE 项，后者在此模式下为常数。网络结构、前向计算和推理参数量保持不变。

两组使用同样的 SNR 交错顺序、数据、学习率 1e-4、批大小和两轮预算。继续采用全部 SNR 零退化验收，B 起点始终为候选。运行结束检查非末层权重逐元素不变，并检查每轮验证的 CSI NMSE 和覆盖范围与起点完全相同。`summary.json` 和 `validation_ber_snr.png`、`validation_nmse_snr.png` 只报告验证集可行性；没有通过验收的改善就保留 B，不扩展训练预算。这个诊断不构成新的独立测试或跨种子结论。
