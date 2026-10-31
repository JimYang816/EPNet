# ICR-MMSE 训练性能与 batch 选择

优化保留当前网络结构、完整信道作用、CSI 修正空间、三阶段冻结/反馈停止规则、逐层损失、QPSK 判决和完整矩阵 NMSE。数据集与物理记录划分保持不变。默认精度仍为 `complex64/float32`，不启用 AMP、不改变 TF32 设置、不缩减 DPSS 基或延迟支撑。

## 实现

1. `dense_device` 将原稠密预处理的双精度基×系数乘法、抽头矩阵装配和二维 FFT 移到设备，最后才转换为模型精度。均衡器和 CSI 修正仍使用完整稠密矩阵。`dense` 保留原 CPU 重建作为对照。`dpss` 是单独保留的实验算子后端，原生单精度严格门槛未通过时不得当作已验证的默认替代。
2. `DPSSReference` 只进入监督接口，保存参考矩阵完整能量、参考矩阵对已存基函数的内积以及修正带内元素。投影用于计算内积，并不把参考矩阵近似为 DPSS 重建。
3. 同一物理窗口的完整参考矩阵（稠密路径）或参考统计量（DPSS 路径）使用有界 RAM LRU 缓存，不写入数据文件或检查点。缓存键包括数据路径、模型身份、窗口、D 块和精度；统计量还包含修正带宽。
4. 原始信道加权列能量在一次前向内复用，各层仍计算当前带状修正的交叉项。训练关闭仅用于诊断的额外残差计算。
5. BER 的整数计数和损失/NMSE 的批级累加留在 GPU，epoch 结束统一读取。保留逐步非有限损失检查和梯度裁剪/有限范数检查。
6. 独立后台线程拥有自己的 HDF5 reader，按原顺序准备有限数量的 CPU batch。CUDA 路径使用 pinned memory 和异步复制；监督字段与推理输入始终分开。异常会传回主线程，退出时回收读取线程。

完整误差仍按以下恒等式计算，其中修正量只在既有修正带内非零：

\[
\|H_0+\Delta H-H_{\rm ref}\|_F^2
=\|H_0-H_{\rm ref}\|_F^2
+2\operatorname{Re}\langle H_0-H_{\rm ref},\Delta H\rangle_F
+\|\Delta H\|_F^2.
\]

第一项包含完整参考矩阵的子空间外能量。NMSE 仍逐块除以完整参考矩阵能量，再平均线性 NMSE；画图时才转为 dB。

## 补偿 CSI 的参考缓存

补偿输入保持 `H_ref + alpha * (H_raw - H_ref)`，其中 `H_raw` 仍用原有双精度 DPSS 抽头展开和 FFT 重建。参考矩阵包含 DPSS 子空间以外的分量，不能直接投影成原 DPSS 系数而声称输入完全相同。

`dense_device` 的补偿输入与监督现在共用同一个参考 LRU。保存的 `complex64` 参考值只缓存一份，送入设备后精确提升为 `complex128` 参与补偿，最后转换为模型精度；没有降低 FFT 或混合的计算精度。同一批参考仅执行一次 pinned-memory 复制及一次主机到设备传输。D1/D2 共用压缩 HDF5 chunk 时一次解压两块，每个缓存条目拥有独立存储，逐块淘汰仍遵守字节上限。缓存不足时正常逐块读取，不改变数据。

这修复了补偿输入曾绕过监督缓存、跨 SNR 和 epoch 重复解压大矩阵的性能问题。完整流程的误差覆盖率准备也使用配置的设备重建和参考缓存。独立正式评测的 `dense` 推理计时入口保持原设置。

以下入口对冻结的旧版管线与当前管线使用相同训练/验证记录：所有 SNR 的输入与监督要求逐元素一致，另核对 A/B/C 损失、梯度、完整 NMSE、硬判决以及连续 Adam 更新后的 BER。计时包含训练和验证；首次冷遍历单列，后续取中位数，不读取 test，不保留诊断训练权重。

对照诊断单独启用 CUDA 确定性算法和对应 cuBLAS workspace。默认 CUDA 卷积反向中的浮点归约，即使旧版自身重复运行也会使连续 Adam 权重逐渐分离；不能据此把少量判决差异归因于缓存。确定性设置仅用于诊断，正式训练的运行设置保持原样。小样本计时需结合正式恢复后的完整 epoch 耗时判断。

```bash
.venv/bin/python -m scripts.evaluate.icr_compensated_cache \
  --baseline-ref b8c275f --device cuda --frames 8 --repeats 2 \
  --output outputs/runs/icr_mmse_compensated/_control/cache_equivalence.json
```

已有补偿训练检查点可在训练进程退出后，用 `scripts.run.icr_cache_upgrade --run ... --verified-report ...` 执行一次有记录的性能迁移，再按原配置 `--resume`。迁移必须验证报告通过、当前指纹一致、旧检查点仅两个缓存模块指纹不同，且 CSI 视图为 compensated、后端为 dense_device；模型、损失等其他源码改变时拒绝迁移。原检查点先完整备份，保留权重、优化器、随机状态、已完成轮次及数据/训练控制身份。常规续训仍执行完整身份校验。

## 测量入口

在服务器项目目录使用已验证的当前结构检查点，例如：

```bash
.venv/bin/python -m scripts.evaluate.icr_training_perf \
  --channel NOF1 --device cuda \
  --operator-backend dense_device --reference-cache-mib 4096 \
  --checkpoint outputs/runs/当前实验/nof1/icr_mmse.pt \
  --output outputs/runs/icr_training_performance_20261028/nof1 \
  --batch-sizes 2 4 8 16 32 64 128 \
  --frames 128 --validation-frames 4 --repeats 3 --threads 4
```

对 NCS1 单独运行相同命令。省略检查点时使用相同种子的诊断权重，报告明确标记，不能用作收敛后算法性能结论。

- 先在 validation 上核对原稠密路径与候选路径的逐层概率、CSI 增量、损失、A/B/C 参数梯度和硬判决，再生成全 SNR 的 BER/NMSE 对照。梯度探针覆盖首、中、末帧和低、中、高 SNR。
- 每个阶段、实现方式、batch 在独立进程中测量。一次完整冷启动遍历单列，之后报告三次完整热遍历的中位数。
- 热遍历包含真实 train 数据读取、监督、前向、损失、反向、梯度裁剪、Adam 更新和 epoch 指标。三阶段使用相同均匀抽样的训练帧、中间 SNR 和 D1/D2，不丢尾批。
- PyTorch allocated/reserved 峰值包括冷启动、共享核、监督、反向与优化器状态。另在热轮期间用 `nvidia-smi` 抽样设备显存、利用率和功耗；设备值包含 CUDA 上下文及其他设备使用者。
- 稠密候选的分步对照是 `dense_reference`、`dense_cached`、`dense_device`、`prefetch`；DPSS 实验候选是 `dense_reference`、`dpss_full_reference`、`compact`、`prefetch`。完整 batch 扫描使用最终预取路径。
- 只读取 train/validation，不利用 test 选择 batch，也不保存计时产生的训练权重。

结果目录包含 `comparison.json`、`equivalence.json`、逐项计时 JSON、`ber_snr.png`、`nmse_snr.png` 和 `batch_speed_memory.png`。

选择规则为：全部 A/B/C 阶段均运行成功、留出至少 2 GiB 设备显存，然后在三阶段合计耗时距离最快不超过 5% 的候选中选择最小 batch。增大 batch 会改变优化轨迹；保持算法和指标公式等价，不代表不同 batch 的长时间训练权重逐项相同。测量对应当前 8 层、修正半带宽 16 的配置；改变网络规模后应重新测量。

## RTX 4080 实测选择（2026-10-28）

测量提交 `c5922b4`，PyTorch `2.5.1+cu121`，RTX 4080 16 GiB。两信道各使用 128 个均匀训练帧、15 dB、D1/D2；每个阶段/批量独立进程，一次冷遍历后取三次热遍历中位数。计时包含数据和监督准备、反向、裁剪、Adam 与指标。

| batch | NOF1 C 阶段 D 块/s | NCS1 C 阶段 D 块/s | 六种信道/阶段最大 allocated（GiB） | 最大 reserved（GiB） |
|---|---:|---:|---:|---:|
| 2 | 53.92 | 54.51 | 0.229 | 0.281 |
| 4 | 92.43 | 91.92 | 0.417 | 0.432 |
| 8 | 127.75 | 128.73 | 0.783 | 0.832 |
| 16 | 132.49 | 133.01 | 1.515 | 1.646 |
| 32 | 130.50 | 131.05 | 2.989 | 3.236 |
| 64 | 125.50 | 126.23 | 5.922 | 6.414 |
| 128 | 113.44 | 114.07 | 11.810 | 12.699 |

两信道分别按规则都选择 **batch=8**。合并六种信道/阶段，batch8 比最快 batch16 的总耗时增加约 3.70%，allocated 峰值约减半。batch8 的 C 阶段平均利用率约 91%，设备采样显存峰值约 1233 MiB，后者包含 CUDA 上下文和桌面进程。

原稠密路径与 `dense_device` 的固定权重验证保留原 `5e-4` 门槛，覆盖首/中/末验证帧、低/中/高 SNR 和 A/B/C。NOF1 最大概率绝对差约 `9.83e-6`、梯度相对差约 `6.37e-5`、完整 NMSE 相对差约 `2.63e-8`；NCS1 对应输出/梯度/NMSE 为零差异。两信道全 SNR 的均匀验证子集最终比特分歧均为零。以上是同权重数值核对，不能当作重新训练后的收敛质量结论。

实测选择已统一写入 `configs/training/icr_mmse.json`：完整稠密设备预处理、batch8、预取2批、RAM参考缓存12288 MiB，模型/损失/学习率/划分和评测设置保持一致。两个验收数据集各有525个训练窗口与75个验证窗口，每个信道的完整双D块参考缓存总计9600 MiB；4 GiB可容纳本次128帧扫描的参考工作集，完整训练跨SNR复用全部参考矩阵需要更大的有界主存缓存。12 GiB是主存上限，缓存按需填充。

这里的 9600 MiB 是**每个信道**在 `complex64` 下的 train+validation 工作集，跨 SNR 复用同一物理窗口参考矩阵；两个信道依次训练，各自建立缓存。`complex128` 下同一工作集为19200 MiB。实际进程还需容纳输入、预取及其他运行内存，选择缓存上限前应检查可用主存。

| 配置项 | 当前值 |
|---|---|
| `training.batch_size` | 8 |
| `training.operator_backend` | `dense_device` |
| `training.prefetch_batches` | 2 |
| `training.reference_cache_mib` | 12288 |
| `output` | `outputs/runs/icr_mmse_compensated` |

`evaluation.batch_size` 和 `budget_match.batch_size` 均为2。`dense_device` 用于训练数据准备；完整流水线和独立评测默认使用 `dense` 推理算子。所有训练数据管线设置均进入续训身份，变更后应使用新的运行目录。

```bash
.venv/bin/python -m scripts.train.icr_mmse \
  --config configs/training/icr_mmse.json --device cuda --threads 4

# 完整验证选型、单种子及消融，按实验目的选择此入口
.venv/bin/python -m scripts.run.icr_mmse \
  --config configs/training/icr_mmse.json --device cuda --threads 4
```

两入口默认依次运行 NOF1、NCS1，均支持 `--channel` 限定信道和 `--resume` 恢复各自的运行。独立训练还支持用 CLI 覆盖上述四项训练设置，`--prefetch-batches 0` 关闭后台预取；完整流水线通过配置文件设置这些项。详细运行目录和命令见 [主说明](icr_mmse.md#6-验证与运行)。

训练性能测量入口 `scripts.evaluate.icr_training_perf` 只从 `--config` 读取数据路径与模型结构；后端、预取和参考缓存由该入口自己的 CLI 参数决定。测完整训练工作集时需显式指定 `--reference-cache-mib 12288`；前述4096 MiB命令用于128帧扫描。

省略 `--config` 时使用上述统一配置。训练输出由配置的 `output` 决定。性能测量曾输出到 `outputs/runs/icr_training_performance_20261028/`；历史产物已从活动目录清理，可从服务器或本地清理备份读取，不提交Git。

同 batch2、C阶段、32训练帧的独立分步基准如下（各自冷遍历后取三次热遍历中位数）：

| 路径 | NOF1 D块/s | NCS1 D块/s |
|---|---:|---:|
| 原同步稠密训练 | 7.77 | 7.69 |
| 缓存完整参考、复用列能量及聚合GPU指标 | 11.51 | 11.63 |
| 进一步将双精度稠密重建移到GPU | 49.78 | 49.21 |
| 进一步加入后台预取 | 53.82 | 53.59 |

完整525训练帧的补测使用batch8、C阶段、15 dB、D1/D2共1050块、12288 MiB缓存，一次冷遍历和一次热遍历：

| 信道 | 冷遍历（秒） | 热遍历（秒） | 热轮 D块/s | 主存参考缓存（GiB） | 进程峰值RSS（GiB） |
|---|---:|---:|---:|---:|---:|
| NOF1 | 41.24 | 8.08 | 129.93 | 8.203 | 10.117 |
| NCS1 | 41.42 | 8.06 | 130.32 | 8.203 | 10.118 |

两次完整帧补测均为冷轮1050次缓存未命中、热轮1050次命中，没有热轮重新读取参考矩阵。显存峰值保持allocated约0.783 GiB、reserved约0.832 GiB。这项补测验证全部训练窗口的缓存工作集；正式跨SNR训练还会逐SNR执行原有更新，不把中间SNR计时当作完整七SNR训练的收敛实验。
