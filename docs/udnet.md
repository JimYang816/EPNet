# UDNet 实现与补偿数据适配

## 依据与范围

依据 Hao Zhao 等《Model-Driven Based Deep Unfolding Equalizer for Underwater Acoustic OFDM Communications》，IEEE TVT 72(5), 2023，DOI `10.1109/TVT.2022.3230143`，式（13）—（15）、图 5 和滑动结构实现。核对的本地论文为 `D:/SCI/paper/EPNet/UDNet.pdf`，SHA256 为 `fdd19460d502fe01f4be8abf8cd45199e99f64a42d62c9601e69fe5beb252eee`。

本实现采用用户确认的补全约定：NOF1、NCS1 各训练一个模型，联合本信道全部 SNR，D1/D2 共用参数。论文中的固定 25 dB 训练和原始星座编号不照搬；本项目使用下述既定数据、200 epoch 上限、20 轮早停和既有比特映射。README、EPNet 实现、四基线注册表和统一比较入口不修改。

## 物理窗口与输入

`src/data/udnet.py` 完成无标签预处理，调用现有 `src/evaluation/inputs.py:prepare_batch` 重建既有补偿 CSI 与接收观测，并在**完整频率范围**先消除所有已知导频的估计贡献。原始文件、补偿系数、划分、CSI 和观测不改写，不再增加补偿或不确定性膨胀。

在 FFT 的物理位置上，以中心块起点 `0,16,32,...` 构造窗口 `起点-8 ... 起点+23`，共 32 个连续复载波。仅保留中心 16 个位置中至少含一个数据槽的窗口。FFT 边界采用零填充，不循环绕回；边界之外的协方差填单位阵且与真实行无互相关。正式数据的边缘保护带已覆盖此上下文。

保留每个载波的真实位置，导频、保护空槽、FFT 边界填充均不压缩。局部矩阵选取同一窗口的接收行和发射列，仅数据列作为未知量。允许截断窗外数据导致的 ICI。正式 FFT1024、512 数据槽对应 56 个有效窗口；中心输出精确覆盖各数据载波一次，训练窗口重叠上下文则按各窗口出现次数参与损失。

令局部单位 RF 噪声协方差为 `C`，Cholesky 分解 `C=L Lᴴ`，以 `W=L⁻¹` 同时白化局部 `H` 和已消除导频的 `y`：

```
A = W H_local[:, data_columns]
b = W y_local
G_raw = Aᴴ A
u_raw = Aᴴ b
power = trace(G_raw).real / number_of_active_data_columns
G_complex = G_raw / power
u_complex = u_raw / power
x_zf = pinv(A, rtol=1e-12) b
```

不乘 RF 方差的倒数，不依赖标签、SNR 或跨样本统计量。白化后，Gram 和 matched 共同除以本窗口有效数据列的平均能量；非数据列不进入均值分母。零能量窗使用尺度 1，得到零 Gram、matched 和 ZF，避免除零。该变换等价于同时缩放 A、b，保持 ZF 解和局部相对信道结构不变。仅对有效数据列计算伪逆，随后散回 32 个物理槽，其余置零。白化、功率归一化、Gram、matched 和 ZF 均以 complex128/float64 计算，缓存及网络使用 FP32。实数形式遵循论文式（8）、（9）：

```
G = [[Re(G_complex), -Im(G_complex)],
     [Im(G_complex),  Re(G_complex)]]
u = [Re(u_complex), Im(u_complex)]
s0 = [Re(x_zf), Im(x_zf)]
```

因此 `G` 是 64×64，`u`、`s0` 均为 64 维，没有现有 EP 实数似然中的系数 2。所有非数据坐标逐层保持为零。

## 网络、判决与损失

`src/algorithms/udnet/model.py:UDNet.forward(gram, matched, zf, data_mask)` 是无标签模型接口，返回 `[batch,22,32,4]` 概率。22 层参数完全独立，共 341,792 个可学习参数；每层由 `Linear(192,48)`、Tanh、`Linear(48,128)` 和按载波的四类 Softmax 构成。

```
t_m = concat(s_m, u, G s_m)             # 192 维
v_m = tanh(W_m1 t_m + b_m1)            # 48 维
v_m += v_(m-1)                         # 仅第二层起，累积的残差状态
p_m = softmax(reshape(W_m2 v_m+b_m2,32,4), dim=class)
p_m = normalize(p_m + p_(m-1))         # 仅第二层起，每槽重新归一化
p_m[non_data] = 0
s_(m+1) = [Re(E_constellation[p_m]), Im(E_constellation[p_m])]
```

首层两处均无残差。λ 固定为 1，其作用吸收入自由全连接权重；不新增可学习 λ。层间使用概率加权星座期望，不使用 argmax，以保持可微。非数据槽的概率和为 0，数据槽概率和为 1。最后一层按四分类 argmax 判决，并只拼接中心 16 个物理位置中的数据槽，顺序与既有数据索引一致。

既有 QPSK 顺序为 `class=2*bit_real+bit_imag`：

| 类 | 比特 | 单位能量星座 |
|---|---|---|
| 0 | 00 | (1+j)/√2 |
| 1 | 01 | (1-j)/√2 |
| 2 | 10 | (-1+j)/√2 |
| 3 | 11 | (-1-j)/√2 |

标签仅在独立的 `src/training/udnet_supervision.py` 和 `src/evaluation/udnet.py:score_udnet` 读取。损失取所有 22 层、整窗全部有效数据槽的交叉熵总和，再除以本批有效窗口标签数；**不除以层数**。统计整轮损失时按各批有效标签数量加权，避免最后小批次或不同掩码带来的偏差。验证采用同一损失。推理适配返回现有 `EqualizerResult`，不会调用 `labels()` 或 `audit()`，不读取真实比特或类别。

## 训练、缓存和恢复

默认配置 `configs/training/udnet.json` 固定 Adam、学习率 0.001、种子 20261027、最多 200 epoch；验证损失严格改善才更新最佳模型，连续 20 轮未改善即早停。每轮用独立的、可恢复的随机生成器对本信道**所有 SNR、D1/D2 和训练帧的全部窗口**统一洗牌。测试集文件不在训练阶段打开，也不参与身份计算或模型选择。

每轮在完整验证窗口上记录分 SNR 的逐层交叉熵、最后一层交叉熵和中心输出 BER。第 1 轮、每 10 轮以及训练结束时，从训练缓存均匀选取至多 256 个固定窗口，记录每层 Tanh 的饱和比例（激活前绝对值大于 3）、平均导数和隐藏层权重梯度范数；探测不更新参数、优化器、梯度缓冲或随机状态。

2026-10-27 的输入归一化修复将检查点及特征缓存版本升级为 v2。旧版输入在 NOF1 上的 Gram 可达百万量级，使 Tanh 饱和；NCS1 的信道统计量则约为万分之一量级。新输入对 H、y 的共同接收增益缩放保持不变。归一化只作用于 UDNet 派生特征，不改写已验收的数据、划分或基线。旧检查点不允许用于新版预处理，须从头训练；新默认目录为 `outputs/runs/udnet_normalized_20261027`，旧实验目录保留供结果对比。

正式训练 525 帧、验证 75 帧，每个信道共 411,600 个训练窗口和 58,800 个验证窗口。一次性缓存 `gram/matched/zf/data_mask/labels` 为独立 NPY 文件，避免反复重建信道及求伪逆。特征键绑定原始数据 SHA256、预处理参数、实现及依赖代码指纹、计算设备、物理窗口和可选 smoke 帧上限；各缓存文件也有 SHA256，损坏或依赖变化则重建。训练缓存约 7 GB，验证约 1 GB，逐信道加载主存；测试数据流式处理，不缓存局部特征。

GPU 使用真实训练窗口，按 2000、1024、512、256 顺序测试全部候选。每个候选用模型副本进行两次完整 Adam 预热，再测一次前向、损失、反向及更新，计入优化器状态。取峰值保留显存不超过预检可用显存 80% 的最大候选。探测不修改正式模型或随机状态；`resource_probe.json` 保存空闲/总显存、预算、每个候选实测峰值及耗时和最终选择。正式运行省略 `--batch-size`，显式覆盖仅供本地 smoke 使用。FP32 网络关闭 CUDA TF32。

每轮以原子替换保存：

- `best.pt`：验证损失最低轮的模型及该轮完整状态。
- `latest.pt`：最新完成轮的模型、Adam 状态、Python/NumPy/Torch/CUDA/洗牌随机状态、早停计数、最佳轮、完整历史、训练身份和实际批次配置。
- `history.json`、`training_summary.json`、`resource_probe.json`、`feature_sources.json`：可独立检查的训练记录、运行配置和来源指纹。

恢复必须使用相同数据、模型、训练配置、实现及依赖代码和数值环境；不匹配明确拒绝，避免混用旧状态。恢复边界是完整 epoch；中途终止则重跑未保存的当前 epoch，随机状态从上一轮恢复。已完成的恢复调用验证身份后直接返回。`stop_after_epochs` 只用于测试模拟中断，不进入正式 CLI 或训练身份。

源码指纹保持 Python 3.10 的 AST 表示，显式保留空列表并忽略新版本新增的空 `type_params`，使服务器 Python 3.10 训练的检查点能在本地 Python 3.13 校验并推理。该兼容处理不改动模型、特征或既有结果缓存键。严格恢复训练仍需切回检查点 `commit` 记录的原始训练版本，并使用其数值环境；兼容修复不会覆盖检查点中的原始训练身份。

## 独立入口与正式运行

从项目根目录运行。训练依赖沿用 `pip install -e ".[learning,test]"`，本机测试采用隔离 `.venv` 和 CPU 版 Torch。

```bash
# 独立训练，两信道分别保存到 nof1 / ncs1
python -m scripts.train.udnet --channel NOF1 --device cuda
python -m scripts.train.udnet --channel NCS1 --device cuda

# 恢复：配置、代码、数据必须与原运行一致
python -m scripts.train.udnet --channel NOF1 --device cuda --resume

# 独立完整评测
python -m scripts.evaluate.udnet --channel NOF1 --device cuda

# 从现有数据执行无标签推理，输出物理数据槽的硬判决
python -m scripts.infer.udnet --checkpoint outputs/runs/udnet_normalized_20261027/nof1/best.pt \
  --dataset data/processed/nof1_compact_20261026_final/test.h5 \
  --snr 10 --frames 0,1 --d 1 --device cuda --output outputs/runs/udnet_inference.npz

# 也可输入无标签充分统计量，输出逐层概率和最终类别
python -m scripts.infer.udnet --checkpoint outputs/runs/udnet_normalized_20261027/nof1/best.pt \
  --features inputs.npz --output outputs/runs/udnet_probabilities.npz

# 正式完整流程：NOF1 训练和评测完成后，再进行 NCS1
python -m scripts.run.udnet --device cuda --threads 4
```

充分统计量 NPZ 仅要求 `gram`、`matched`、`zf`、`data_mask`。现有 `python -m scripts.evaluate.compare` 和 EPNet 的独立入口继续按原方式工作。

独立评测调用现有四基线评测器，由该评测器重新校验数据、输入构造、算法、Robust EP 训练协方差、评分、环境及 NMSE 依赖，检查每个缓存块的校验和。仅依赖匹配的部分复用，不根据旧报告中的曲线直接认定缓存可用。UDNet 结果单独按最佳模型 SHA256、输入/模型/评分代码和测试数据身份缓存。正式评测强制完整测试集全部 SNR、600 帧、D1/D2 和 512 数据槽，每 SNR 每算法 1,228,800 比特、UDNet 614,400 符号。

每信道 `evaluation/` 输出 `metrics.json`、`report.html`、五算法 `ber_snr.png`、共享输入 CSI `nmse_snr.png`、UDNet `ser_snr.png`、`training_curves.png` 和分 SNR 验证过程图 `validation_by_snr.png`。NMSE 定义与现有比较完全相同：各帧各 D 的全矩阵线性相对平方误差先平均，再转 dB；它是各算法共享的输入质量指标。图中零错误按 0.5/总数显示并说明。

正式流程逐信道完成后删除该信道的可重建特征缓存，保留检查点、历史、实际配置、指标和来源指纹。共享基线/结果缓存继续供既有比较复用，完整原始数据留服务器，任何数据和运行产物均不提交 Git。服务器按本地已测试并推送的提交拉取，重新检查 GPU/内存/磁盘后运行。最佳模型、完整恢复状态、报告、指标、配置及图表经 SSH 回传本地。

## 验证

```bash
python -m pytest tests/algorithms/test_udnet.py tests/training/test_udnet_training.py -q
```

测试覆盖真实物理载波槽、首尾零填充、中心无遗漏无重复覆盖、非数据槽隔离、双精度白化与缩减数据列伪逆、无 SNR 缩放、功率归一化、零信道和跨八个数量级接收增益的特征一致性与有效梯度、既有 QPSK 类别、首层与后续层残差公式、概率和与概率期望反馈、全部 22 层梯度、损失分母与分 SNR 指标权重、缓存损坏重建、恢复前后逐参数和验证指标精确一致、旧预处理检查点拒绝、数据/配置身份拒绝、HDF5 标签访问拦截，以及合成数据上的独立五算法评测与缓存复用。
