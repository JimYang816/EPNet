# ICR-MMSE-Net：联合 CSI 修正与 MMSE 深度展开均衡

本文是当前修订方案，主实验采用 **compensated CSI 视图、按矩阵元素对齐的反馈、A/B/C 三阶段训练**。

主配置采用 `model.noise_mode="diagonal"`：保留现有数据及各频点噪声功率，用单位 RF 噪声协方差的实对角近似完整协方差。`"full"` 保留完整协方差求解，供同权重对照；模型构造器未指定该字段时仍采用 `"full"`，避免缺省配置改变含义。检查点记录实际模式，原有代码身份校验继续生效。
默认数据由用户固定：SNR 0→30 dB 时，NOF1 平均 NMSE 为 −12.70→−18.30 dB，NCS1 为 −6.20→−10.67 dB。沿用已验收数据及原划分；未经用户明确要求不得更换为 raw CSI 或修改目标。
研究目标是在 NOF1、NCS1 上对比 MMSE 基线、UDNet 与 ICR-MMSE-Net 的 BER，同时验证全矩阵 CSI 改善并比较效率。
配置和代码本身不代表目标已经达到；本地仅做小规模 CPU 验证，正式结论来自服务器上的完整实验。

ICR-MMSE-Net 默认 `model.precision="complex64"`：复数物理算子、CSI、观测、噪声协方差和监督张量采用 `complex64`，实数特征、RF 方差及网络参数采用 `float32`。
原 `dense` 路径在 CPU 以双精度重建完整 CSI，再转换为模型精度并传输到设备；RTX 4080 配置使用训练路径 `dense_device`，将双精度基×系数、完整矩阵装配和二维 FFT 放在设备上执行，完成重建后再转换为模型精度。
因此 `dense_device` 的重建阶段包含双精度设备暂存，后续均衡、CSI 修正与网络参数仍采用配置精度。
ICR 输入适配不预先消除导频，由模型按当前 CSI 重算；模型边界也按配置转换，避免调用者传入双精度后静默恢复双精度计算。
需要数值校验时显式设置 `model.precision="complex128"` 或训练/完整运行命令添加 `--precision complex128`，对应实数计算及网络参数为 `float64`。
精度属于检查点和续训身份，预热模型须匹配精度。各报告记录实际计算精度；传统基线仍沿用其已验证的 `complex128/float64` 实现。

算法公式与参数量见 [算法设计](icr_mmse_algorithm_design.md)，后端说明见 [DPSS 后端](dpss_backend.md)，RTX 4080 的 batch 与缓存实测见 [训练性能](icr_mmse_training_performance.md)。

## 1. 输入视图与监督边界

`CompactDataset(...)` 默认采用 `input_view="compensated"`。按已保存的 alpha 构造完整的
`H_init = H_ref + alpha * (H_raw - H_ref)`。启动时校验目标曲线和 alpha 对应的平均 NMSE，发现与用户固定值不符即拒绝运行。
该数据是已验收的参考辅助构造。ICR 模型只接收构造后的完整 CSI、含噪观测、已知导频、物理 FFT 位置及掩码、RF 方差和单位接收噪声协方差；不单独接收参考 CSI、alpha、真实误差或标签。

`src/data/icr_mmse.py:input_batches()` 在完整载波网格上生成 D1/D2 输入。
`supervision()` 独立读取标签与参考 CSI；阶段 A 训练通过 `include_reference=False` 只读取符号监督。
模型 forward 不接受标签或参考参数。参考信道参与既定数据构造，并可用于训练监督及验证/测试评分；结果必须标记其参考辅助构造属性。

Single-tap MMSE、Full-matrix LMMSE、UDNet 和 ICR-MMSE-Net 均使用同一 compensated 视图。主评测不再拟合或加载 EP 专用误差协方差。
UDNet 特征、ICR 检查点和评测缓存均绑定输入视图及代码/数据身份，不复用 raw 实验产物。旧 raw 训练评测已停止、产物已清理；数据中的 raw 字段保留供显式诊断。
`dense` 和 `dense_device` 均先在双精度中完成完整补偿构造，再转换为模型精度。单独 DPSS 乘积算子无法表示参考 CSI 的子空间外分量，不支持 compensated 主实验；应选择两个完整矩阵路径。

## 2. 联合层与确定性反馈

完整信道保存为基础矩阵加带状增量：

\[
H_{t+1}=H_t+\eta_t\mathcal P_B(g_t\odot D_t),\qquad
\eta_t=\operatorname{sigmoid}(a_t)\in(0,1).
\]

NOF1、NCS1 的修正半带宽统一固定为 `B=16`，对应主对角线及两侧各16个物理子载波，共33个偏移。
带外的原始 CSI `H_init` 元素始终保留。`BandChannel` 提供完整乘法、伴随、列能量与全矩阵误差能量；
通过基础矩阵与带内交叉项计算，无须逐层生成完整新矩阵或 Gram。

每轮先构造完整软状态：数据位置为上一轮均值/方差，导频为已知值/零方差，空槽为零/零方差。
数据初值始终是均匀 QPSK 先验，即均值 0、方差 1。随后计算完整残差 `r = y - H_t @ xbar`。
对每个接收行 `k` 和偏移 `d∈[-B,B]`，取物理列 `j=k+d`：

\[
F_t[k,d]=[H_{init}[k,j], H_t[k,j],
\bar x_j,v_j,r_k\bar x_j^*,H_t[k,j]v_j,
M_{data,j},M_{pilot,j},M_{valid}[k,d]].
\]

索引、共轭与乘积由 `build_feedback_features()` 确定性计算。
每偏移保留 **14 个实数特征**：初始 CSI 实/虚部、当前 CSI 实/虚部、软均值实/虚部、
方差、残差乘共轭均值实/虚部、CSI 乘方差实/虚部，以及数据/导频/有效性掩码。
残差实/虚部每个接收行只保存一次；空槽掩码由 `null = valid - data - pilot` 恢复。
越界的偏移特征全部置零，不循环折返，不压缩数据载波索引；行级残差不随偏移复制。
原先逐偏移的残差可由行级残差与有效性掩码相乘恢复，因此边界信息仍完整。
`CSIFeedback.offsets` 的形状为 `[batch,N,2B+1,14]`，`CSIFeedback.residual` 为 `[batch,N,2]`。
全部偏移拼成通道后，加上行级残差的 2 维及归一化 RF 方差、单位噪声功率、载波位置的 3 维，得到 `14(2B+1)+5` 个输入通道。
默认 `B=16` 时输入为 `[batch,467,1024]`；1024 点 FFT、物理载波位置和两项物理乘积均保留。
复数观测与 CSI 采用相同的输入功率归一化，输出增量恢复到原物理尺度。

共享修正网络严格固定为 `Conv1D(32,k=3)→SiLU→Conv1D(32,k=3)→SiLU→Conv1D(3(2B+1),k=1)`。
输出分别是 `2(2B+1)` 个线性修正量及 `2B+1` 个 Sigmoid 门控。
仅修正量输出头零初始化，保持最初行为不修正。感受野为 5 个接收行；每行已含所有偏移反馈。
物理乘积是启发特征，不宣称其差是有色噪声模型的精确梯度。

默认修正网参数量由原 564 通道布局的 60,547 降为 **51,235**；对一个 1024 载波 D 块，
每轮修正 CNN 的理论实数乘加由 61,833,216 降为 **52,297,728**，减少约 **15.4%**。
八轮共享同一修正网；该比例只描述修正 CNN，端到端耗时仍包含完整矩阵运算及预处理，须单独测量。
新布局保留可恢复的特征信息，但不宣称与旧 CNN 函数等价；第一层权重形状已改变，旧 564 通道检查点不能直接用于新布局，需重新训练。

## 3. MMSE 展开层的具体计算

各联合层使用独立参数的 MMSE 展开模块。更新 CSI 后，重新计算导频消除、完整匹配滤波与正规算子作用：

\[
y_D=y-H_{t+1}x_{known},\quad
b=H_{t+1}^{H}Wy_D,\quad
A\bar x=H_{t+1}^{H}W(H_{t+1}\bar x).
\]

其中 `C` 是单位 RF 噪声协方差。对角模式使用 `W=diag(C)^{-1}`，仅逐元素除法，不执行 Cholesky 分解或三角求解；完整模式使用 `W=C^{-1}`，其 Cholesky 因子在同一模型内跨层、批次、SNR 及训练轮次复用。
完整模式的缓存仅保留一份协方差快照与因子，按内容、设备和精度检查；协方差变化或模型迁移时重新分解，缓存不进入检查点。
RF 方差单独参与正规方程，不触发单位协方差重分解；需要对协方差求梯度时绕过缓存。两种模式的预条件器及 CSI 网络噪声输入均使用相同的 `C` 对角，只改变噪声逆算子的作用。完整信道矩阵及所有观测仍保留。
线性子步是带可学习步长的对角预条件 Richardson 更新：

\[
z=\bar x+\omega_t\frac{b-A\bar x-\sigma_{RF}^2\bar x}
{q+\sigma_{RF}^2},\qquad
q_j=\sum_k |H_{k,j}|^2/C_{k,k}.
\]

这是对 LMMSE 正规方程的可学习迭代近似，不是精确的闭式 LMMSE 求解。
`q` 是预条件器近似，正规作用仍使用全部信道耦合与所选的 `W`，不形成完整 Gram、不逐层求信道逆。
QPSK 距离 logits 配合局部窗口校准产生归一化后验。
窗口默认宽 32、中心 16；中心输出覆盖完整物理网格一次，拼接为下一轮全局软状态。
仅数据位置参与检测与 CE；导频/空槽状态固定。每轮的导频消除、残差、归一化检测特征均跟随当前 CSI 重算。

## 4. 三阶段训练与选择

NOF1、NCS1 分别训练；每个信道全部 SNR 与 D1/D2 共享参数，不打开测试集拟合。

| 阶段 | 可训练参数 | 反馈与损失 |
| --- | --- | --- |
| A | 所有 MMSE 展开层 | 关闭 CSI 修正；均匀初始先验；逐层 CE；保存 `warmup.pt` 固定 CSI 基线 |
| B | 共享修正网络及逐层阻尼 | 冻结均衡器；当前预测 CSI 实时运行均衡器；软反馈 detach；逐层完整矩阵 NMSE |
| C | 全部参数 | 开放软反馈梯度；各层 `CE + lambda_h * NMSE` 的平均 |

阶段 B 不缓存固定 `H_init` 下的软预测复用于各轮。训练及推理的 CSI 更新→均衡→软状态顺序一致。
每阶段恢复验证集最佳参数再进入下一阶段。A 以验证 BER 选择；B 以验证 NMSE 选择；C 优先选择同时改善 CSI 的最低 BER 检查点。
若没有 CSI 改善，保留诊断结果但不得将候选声明为通过选型。

### 同权重噪声近似小规模对照

`python -m scripts.evaluate.icr_noise --dataset <原始数据目录>/test.h5 --checkpoint <检查点> --run outputs/runs/<本次对照> --device cuda --max-frames 16 --timing-repeats 5`

入口从同一检查点复制完全相同的权重，以相同测试帧、SNR、D1/D2 输入分别执行 `full` 与 `diagonal`，不按测试结果拟合参数。默认从测试帧索引范围均匀抽样，以扩大物理记录覆盖；报告保存实际帧索引和分记录统计，`--frame-selection first` 可显式复现前缀抽样。训练入口可用 `--noise-mode full` 建立完整噪声权重的短训参照。报告记录训练模式、代码及数据身份；固定权重切换仅用于测量近似的直接影响，不等价于分别充分训练后的性能比较。

每批原始 CSI 重建与输入转换只执行一次，两种模式交替计时顺序。首批冷启动单独报告；重复前向的总耗时取中位数，再分别加上相同的实测输入准备时间，形成端到端耗时估计。标签读取与评分不计入计时。输出 `comparison.json`、`ber_snr.png`、`nmse_snr.png` 和 `runtime.png`，NMSE 先平均完整矩阵线性值再转换为 dB。有限帧结果标记为小规模诊断，不能用于正式选型或总体等效性结论。
阶段冻结、优化器、随机状态和阶段内轮数写入 `latest.pt`，支持精确续训；数据、代码或视图变化则拒绝续训。

两信道均固定 `B=16`，不参与验证选型；`model.bandwidth` 是唯一带宽配置，不再接受 `selection.bandwidth`。
默认起点 `T=8, lambda_h=0.1`；各阶段默认 10 epoch 是可调整的运行配置，不是已经验证的最佳轮数。
正式流水线先输出训练及验证集在 `B=16` 下的 CSI 误差能量覆盖率，然后仅用验证集依次搜索：

1. `lambda_h ∈ {0.01, 0.1, 1}`。
2. `T ∈ {4, 8, 12, 22}`。

每步以验证 BER 为主，同时要求全矩阵平均 NMSE 优于输入 CSI。选定配置再以 `selection.seeds` 指定的种子重新训练与测试，当前仅使用种子 `20261028`。省略该字段时使用 `training.seed`，不额外运行其他种子。
测试集不用于超参数、轮数、预算或消融配置选择。

## 5. 比较、消融与输出

主比较为 Single-tap MMSE、Full-matrix LMMSE、compensated 视图重新训练的 UDNet、ICR-MMSE-Net，共四种算法。EP、Robust EP 已从算法注册、评测配置、统计对照和效率对比中移除。
固定 CSI 使用同次阶段 A 的 `warmup.pt`。一次修正、逐层修正、去不确定度、去门控从同一预热均衡器开始，
不改变完整基础 CSI、全局算子或物理网格对齐。修正模式支持 `fixed/once/iterative`；其余控制见配置 `ablations`。
预算匹配一次修正使用固定 CSI 下的额外有效均衡更新，不增加参数；预算由验证数据上的全预处理/全迭代实测耗时选择并报告匹配比例。
预算无法达到设定容差时必须标记未匹配，不能只凭相同层数声称公平。

评测保存 `metrics.json`、`ber_snr.png`、`nmse_snr.png`、`layerwise.png`。
包括 BER–SNR、修正前后完整矩阵平均 NMSE–SNR、每层 BER/NMSE、CSI 恶化比例及相邻层反馈 BER 恶化比例。
NMSE 先平均每块的线性全矩阵 NMSE，再转换 dB。
耗时包含原始 CSI 重建、预处理及全部迭代；监督读取和评分单独计时。
同时报告总参数数、CUDA 峰值已分配/保留显存与 OS 进程峰值 RSS；CPU RSS 是进程生命周期峰值，应在独立进程中做严格内存比较。
物理记录是统计单元，使用整条记录 bootstrap；载波、D1/D2、重复载荷和重复噪声不当作独立信道样本。

## 6. 验证与运行

### 配置选择

统一配置为 `configs/training/icr_mmse.json`，已吸收 RTX 4080 实测设置。数据集、compensated 视图和模型采用 `complex64`、`diagonal`、8 层、修正半带宽16、32/16窗口、逐层修正、不确定度和门控开启、每层一次均衡更新。A/B/C各10 epoch，Adam学习率0.001，初始 `lambda_h=0.1`，种子20261028。

| 配置项 | 当前值 |
| --- | --- |
| 输出根目录 | `outputs/runs/icr_mmse_compensated` |
| `training.batch_size` | 8 |
| `training.operator_backend` | `dense_device` |
| `training.prefetch_batches` | 2 |
| `training.reference_cache_mib` | 12288 |
| `selection.seeds` | `[20261028]` |
| `evaluation.batch_size` | 2 |
| `budget_match.batch_size` | 2 |

RTX 4080 配置的参考缓存是按需填充的主存 LRU 上限。当前 `complex64` 下，每个信道的 525 个训练窗口和 75 个验证窗口、各两 D 块，完整参考矩阵合计 9600 MiB；12288 MiB 可容纳这一工作集。切换 `complex128` 后，同一工作集为 19200 MiB，须重新评估主存预算。缓存与预取设置、batch、后端及模型精度都进入训练身份，恢复时须保持一致。

独立训练入口可用 `--batch-size`、`--operator-backend`、`--prefetch-batches`、`--reference-cache-mib`、`--precision` 和 `--noise-mode` 覆盖配置；`--prefetch-batches 0` 关闭后台预取。完整流水线通过 `--config` 选择训练设置，并支持 `--precision` 覆盖；其他训练覆盖项应写入配置。

评测入口的 `--operator-backend` 支持 `dense`、`dpss`，默认 `dense`；`dense_device` 是训练数据准备路径。训练 batch 为8，评测和预算匹配 batch 为2。完整流水线执行验证集 `lambda_h` / 层数搜索、单种子重训与消融；batch 8的吞吐量结论对应8层模型，改变深度后需重新测量。直接调用训练函数时，未提供的数据管线参数仍沿用兼容默认值：batch 2、`dense`、预取2批和512 MiB缓存；正式入口通过JSON传入上表设置。

### 验证与命令

必要测试覆盖以下约束：

- 修改参考 CSI、标签、α、参考投影和实际 SNR 后，固定原始输入与模型的预测不变；输入读取遇到这些字段即失败。
- `xbar[k+16]` 改变对应 `F[k,16]`；共轭乘积、导频、空槽与边界符合显式定义。
- 改变当轮 CSI 会重算导频消除、残差、匹配特征及后续软状态。
- 基础矩阵加带状增量的乘法/伴随/能量与小矩阵显式结果一致；零修正退化为固定 CSI；概率及梯度有限。
- 三阶段参数冻结正确，续训与不中断训练逐参数相等；训练从不打开 `test.h5`。

本地小规模检查：

```powershell
python -m pytest tests/algorithms/test_icr_mmse.py tests/data/test_raw_csi.py tests/training/test_icr_mmse_training.py tests/evaluation/test_icr_evaluation.py tests/evaluation/test_icr_pipeline.py tests/evaluation/test_icr_efficiency.py
python -m scripts.train.icr_mmse --channel NOF1 --device cpu --max-frames 1 --stage-epochs 1 1 1 --batch-size 2 --reference-cache-mib 64 --output outputs/runs/icr_cpu_smoke
```

服务器正式运行前，通过 `ssh lab448` 检查可用 CPU/GPU/显存与磁盘，再在 `/home/fay/JimYang/EPNet` 拉取对应提交：

```bash
# RTX 4080：按配置训练，两信道依次运行
python -m scripts.train.icr_mmse --config configs/training/icr_mmse.json --device cuda --threads 4

# 完整实验：覆盖率、验证选型、单种子及消融
python -m scripts.run.icr_mmse --config configs/training/icr_mmse.json --device cuda --threads 4

# 恢复上述完整流水线
python -m scripts.run.icr_mmse --config configs/training/icr_mmse.json --device cuda --threads 4 --resume
```

独立训练与完整流水线按实验目的选择一个入口。独立训练检查点位于 `<output>/<channel>/icr_mmse.pt`；完整流水线的最终主模型位于 `<output>/<channel>/seed_<seed>/icr_mmse.pt`，各自保存 `latest.pt` 供对应入口续训。省略 `--config` 时使用上述统一配置。
训练与评测输出都放在 `outputs/runs/`；数据、特征缓存、检查点和结果不提交 Git。
ICR-MMSE-Net 的实现位于 `src/algorithms/icr_mmse/`。当前学习算法保留 UDNet 与 ICR-MMSE-Net，独立历史 EPNet 及旧版全 CSI 评测流程已移除。

正式效率比较使用单独入口，四种算法依次在各自新进程中运行。以下以 NOF1 的首个种子为例：

```bash
python -m scripts.evaluate.icr_efficiency \
  --dataset data/processed/nof1_compact_20261026_final/test.h5 \
  --comparison-report outputs/runs/icr_mmse_compensated/nof1/seed_20261028/evaluation/metrics.json \
  --checkpoint outputs/runs/icr_mmse_compensated/nof1/seed_20261028/icr_mmse.pt \
  --udnet-checkpoint outputs/runs/icr_mmse_compensated/nof1/udnet_compensated/best.pt \
  --output outputs/runs/icr_mmse_compensated/nof1/efficiency.json --device cuda --batch-size 2 --threads 4
```

该入口加载各自检查点及热噪声模型，不读取训练数据或测试标签；参考 CSI 和 alpha 只用于既定输入构造。
报告初始化、原始 CSI 重建至全部迭代、摊销每块耗时、参数数和独立进程峰值内存。
进程启动及通用模块导入不计入耗时，数据、检查点及报告的身份校验时间单列，避免文件扫描污染推理速度比较。
内存峰值包含导入开销；OS 文件缓存不主动清空。

### 固定 LMMSE 的 CSI 对照诊断

为隔离 CSI 质量对 BER 的影响，可将 数据集 CSI、已完成候选最后一层输出的完整修正 CSI、真实参考 CSI 分别交给同一个完整矩阵 LMMSE。该入口只读固定验证集和冻结检查点，CPU 独立运行，不修改训练、选型、评测结果或共享缓存。

```bash
python -m scripts.evaluate.icr_csi_lmmse \
  --dataset data/processed/nof1_compact_20261026_final/validation.h5 \
  --candidate-report outputs/runs/icr_mmse_compensated/nof1/selection/lambda_h_0p01/validation/metrics.json \
  --output outputs/runs/nof1_csi_lmmse_20261028 --lambda-h 0.01 --batch-size 2 --threads 4
```

三组使用同一完整验证帧、D1/D2、SNR、接收观测和比特标签；统一以 `complex128`、完整热噪声协方差、全部接收行和全部数据列求解。每次更换 CSI 均重新消除已知导频并创建独立似然缓存。修正网络保持检查点的原始精度与噪声模式，完整输出转换至 `complex128` 后交给 LMMSE；参考 CSI 仅在修正及两组接收机预测完成后，用于 oracle 分支和指标计算。

入口校验候选报告、检查点哈希、`lambda_h` 和数据身份，输出 `metrics.json`、`metrics.csv`、BER–SNR 与平均全矩阵 NMSE–SNR 的 PNG/SVG。NMSE 先对每个 D 块的线性值求平均，再转 dB；真实 CSI 的线性 NMSE 为零，JSON 的 dB 字段为 `null`，图中标记为负无穷。零误码 BER 仅在作图时显示于 `1/total_bits`，数值文件保留零。
