# OAMP-Net2、MMSE-PIC 与七算法统一评测

本页同步 2026-10-30 的训练、评测配置和诊断入口。主对比为 Single-tap LMMSE、Full-matrix LMMSE、
MMSE-DFE、MMSE-PIC、UDNet、OAMP-Net2、ICR-MMSE-Net。MMSE-DFE 的计算与固定检测顺序见 [说明](mmse_dfe.md)。保持已验收数据文件、物理记录划分、
compensated CSI、导频、接收观测、QPSK 映射及 BER 定义不变。
NOF1 的输入平均 NMSE 为 −12.70→−18.30 dB，NCS1 为 −6.20→−10.67 dB，SNR 0→30 dB。

## 1. 固定输入与比较范围

两个新方法调用 `src/evaluation/inputs.py:prepare_batch`：先在完整物理载波矩阵上
用估计 CSI 消除全部已知导频，再保留所有接收行、所有数据列。不做滑窗或带状截断。
使用完整热噪声协方差与样本 RF 噪声方差；不使用真实 CSI 误差、EP 专用统计量或测试集调参。

设导频消除后的模型为 `y = H x + n`，噪声协方差为 `sigma_rf² C`。
令 `A = C^(-1/2) H / sigma_rf`、`b = C^(-1/2) y / sigma_rf`，
则白化噪声协方差为单位阵。共享似然量为 `G = Aᴴ A`、`g = Aᴴ b`。
所有 FFT 接收行均参与计算，因此 OAMP 方差估计减去的噪声能量是接收行数，
而非数据载波数；复数方差不额外乘除 2。

两个方法均固定输入 CSI，仅改善检测。它们与两种 MMSE、UDNet 共用输入 NMSE 曲线；
ICR-MMSE-Net 另画修正后的全矩阵 NMSE。NMSE 始终先按帧、D 块计算完整矩阵
线性相对平方误差、取算术平均，再转 dB。

## 2. OAMP-Net2

依据 He、Wen、Jin、Li，*Model-Driven Deep Learning for MIMO Detection*，
IEEE TSP 68:1702–1715, 2020，DOI [10.1109/TSP.2020.2976585](https://doi.org/10.1109/TSP.2020.2976585)，
[作者全文](https://arxiv.org/html/1907.09439v2)，式 (19)、(24)–(28)、(32)。

每层四个独立可学习实标量 `gamma, theta, phi, xi`，初值分别为 `1, 1, 1, 0`。
默认 10 层，共 40 个参数。使用论文的残差方差估计、去相关 LMMSE 线性估计、
QPSK 后验均值及 `phi * (posterior_mean - xi * r)` 非线性输出。
式 (27) 的 `I - theta W A` 使用 `theta`；线性残差步使用 `gamma`。
方差下限沿用论文的 `5e-13`，检测、预处理及训练均为 `complex128/float64`。

实现以完整 `G = Q diag(lambda) Qᴴ` 的特征分解计算 LMMSE 滤波器及迹，
与逐层直接求逆等价，每个样本仅分解一次。保留所有特征向量和特征值，
仅将数值舍入产生的微小负特征值截为零。标签不进入特征构造或 forward。

本项目适配为固定 compensated CSI 的**检测器**：没有实现论文的完整 JCESD 信道重估外循环，
也没有按真实 CSI 误差扩充噪声协方差。与现有固定 CSI 检测基线使用相同热噪声输入。
与 UDNet 的训练口径一致，NOF1、NCS1 各训练一个模型，联合本信道全部 SNR，D1/D2 共用参数。
这与原论文通常逐 SNR 训练的设置不同，不能直接引用原论文的性能数值。

`configs/training/oamp_net2.json` 默认 Adam、学习率 0.001、200 epoch 上限、20 轮早停、种子20261030。
训练 `batch_size=8`，谱特征准备 `frame_batch_size=8`，独立评测 `batch_size=8`；这里的训练 batch 为完整 D 块，不是 UDNet 的滑动窗口数。
训练与评测均显式指定 `input_view="compensated"`。
损失采用式 (32)：最终层符号向量平方误差之和，再对样本平均；验证用同一损失选最佳模型。
训练只打开并绑定 `train.h5`、`validation.h5`、`model.h5`；不打开或哈希 `test.h5`。
检查训练、验证物理记录不相交。测试时再次检查测试记录与训练、验证记录均不相交。

检查点记录数据/代码/预处理身份、compensated 视图、信道、SNR 网格、数据载波、
优化器、shuffle 与随机生成器状态。使用 `weights_only=True` 加载；不接受 raw、
不同信道、不同输入模型、不同代码或正式评测中的小样本检查点。

完整谱特征保存为实验目录下的只读 mini-batch 内存映射缓存，按块校验和支持中断恢复与损坏重建。
缓存仅限训练/验证，身份绑定视图、内容哈希、代码及数值环境。
512 数据载波时，仅特征向量每 1000 个 D 块约占 3.91 GiB。
当前固定配置每信道包含 7350 个训练 D 块、1050 个验证 D 块，谱缓存合计约 33 GiB，
运行前应预留这些磁盘空间及至少 1 GiB 余量。
完整流水线在评测成功后删除这些可重建特征；`--keep-features` 可保留。

## 3. MMSE-PIC

采用固定 CSI、无信道译码器的阻尼软反馈适配。参考 Matthé、Zhang、Fettweis，
[*Iterative Detection using MMSE-PIC Demapping for MIMO-GFDM Systems*](https://www.vodafone-chair.org/pbls/legacy/m-matthe/Iterative_Detection_using_MMSE-PIC_Demapping_for_MIMO-GFDM_Systems.pdf)，
式 (7)–(12) 的一般线性系统软符号均值/方差、干扰抵消和 MMSE 滤波。
本项目使用 QPSK 精确后验均值，不使用 max-log 译码或 LDPC 外循环，因此不是该文完整编码接收机的复现。

初始化软均值为零、方差为一。每轮利用其他数据符号的软均值抵消干扰，
将其剩余方差计入 MMSE 滤波。目标符号自身的反馈先验从标量似然中消除：

```text
V = diag(v)
K = Aᴴ (I + A V Aᴴ)^(-1) A
u = Aᴴ (I + A V Aᴴ)^(-1) (b - A m)
z_i = m_i + u_i / K_ii
tau_i = 1 / K_ii - v_i
```

对 `z_i, tau_i` 使用均匀 QPSK 先验求后验均值；不再乘入自己的上一轮概率。
实现通过数据维 Hermitian 系统 `I + sqrt(V) G sqrt(V)` 求解，避免逐个目标符号求逆。
反馈均值采用阻尼混合，再由单位 QPSK 能量计算方差 `1-|m|²`。
最后一轮输出该轮后验均值，零信道输出零均值。

默认 10 轮、更新权重 0.5、方差下限 `1e-10`。这些是预设参数，尚未通过正式验证集搜索。
后续如需调参，只在固定验证集选择，然后冻结参数评测测试集。
算法无需训练检查点，支持现有 NumPy CPU / PyTorch CUDA 后端，保持双精度。

## 4. 运行

代码同步到服务器后，先检查 GPU/主存/磁盘资源。使用现有 Python 环境：

```bash
# 新学习型方法：依次训练两个信道，并评测 MMSE、MMSE-PIC、OAMP-Net2
python -m scripts.run.oamp_net2 --device cuda --threads 4

# 单独训练、恢复或评测
python -m scripts.train.oamp_net2 --channel NOF1 --device cuda
python -m scripts.train.oamp_net2 --channel NOF1 --device cuda --resume
python -m scripts.evaluate.oamp_net2 --channel NOF1 --device cuda

# 本地已有数据时的小样本 CPU 检查；单独输出，不能用作正式检查点
python -m scripts.train.oamp_net2 --channel NOF1 --device cpu --max-frames 2 --epochs 1 --batch-size 2 --output outputs/runs/oamp_net2_smoke

# 无需任何学习检查点，运行四种传统方法
python -m scripts.evaluate.compare --device cuda --baselines-only

# 七算法统一评测
python -m scripts.evaluate.compare --device cuda --threads 4
python -m scripts.evaluate.compare --channel NCS1 --device cuda --force mmse_pic
python -m scripts.evaluate.compare --plot-only
```

OAMP 配置为 `configs/training/oamp_net2.json`，沿用现有两个数据集目录。
检查点默认在 `outputs/runs/oamp_net2_compensated/{nof1,ncs1}/best.pt`。
`latest.pt` 支持续训；`best.pt` 只由验证损失严格改善更新。

统一配置为 `configs/evaluation/ber_comparison.json`。其中 `checkpoints` 为每个信道的
UDNet、OAMP-Net2、ICR-MMSE-Net 路径；默认 ICR 路径按现有 `seed_20261028` 流水线填写，
应指向实际选定的正式检查点。统一入口在开始计算前校验全部所需检查点；
缺失或身份不匹配会报错，不会跳过曲线或拿未训练模型替代。
没有 `checkpoints` 的旧自定义配置仍保留原基线接口行为。
统一评测默认 batch 8，四种传统方法的参数由 `algorithms` 提供；独立 ICR 流水线的 batch 2 不会覆盖此值。各入口的算法数量、强制重算和离线绘图规则见 [均衡评测与缓存](equalizer_comparison.md)。

统一结果在 `outputs/runs/equalizer_comparison_mmse_dfe_20261031/{nof1,ncs1}/`，原有六算法结果保留：
`metrics.json`、`report.html`、七条 BER 曲线的 `ber_snr.png`、输入/修正 CSI 的 `nmse_snr.png`。
各子评测保留详细来源。正式测试每个算法、每个 SNR 必须恰好评测 600 帧、D1/D2、
512 个数据槽，即 1,228,800 比特；汇总会再次核对数据哈希、SNR、视图和比特总数。

两种 MMSE 的计算身份保持原样，加入 MMSE-PIC 不会让其有效缓存失效。
MMSE-PIC 的参数、OAMP 的检查点 SHA256 和各自计算依赖进入结果缓存身份。
OAMP 逐 SNR/帧/D 块缓存，可跨 batch 大小复用并校验损坏；`--force all` 同时刷新 OAMP。
UDNet 保留已有缓存行为，ICR 保留已有完整重评行为。
离线绘图使用保存数值，不需要数据、检查点或 GPU。

效率入口 `scripts.evaluate.icr_efficiency` 默认在七个独立新进程中测量七种方法，
必须提供 `--checkpoint`、`--udnet-checkpoint` 和 `--oamp-net2-checkpoint`。
MMSE-PIC 使用统一报告中的实际迭代参数。OAMP 的逐样本完整谱分解计入预处理时间，
不复用训练谱特征或 BER 结果缓存；计时进程禁止读取测试标签。

## 5. OAMP-Net2 数值核验与独立续训诊断

`scripts.evaluate.oamp_net2_diagnostics` 只使用已有训练、验证划分，从正式 `best.pt` 启动独立诊断，不读取 `test.h5`，不覆盖原检查点或测试报告。先核对高 SNR 验证样本的四组标量梯度与接收空间直接求逆参考；误差超过代码门槛即停止。随后记录原模型、未训练 OAMP 和传统方法的验证结果，并从同一原模型及 Adam 状态分别运行：

| 试验目录 | 训练范围 | 默认追加 epoch | 学习率 | 选模依据 |
|---|---|---:|---:|---|
| `joint_lr1e-3` | 全部 SNR | 50 | `1e-3` | 完整验证集向量 MSE |
| `joint_lr1e-4` | 全部 SNR | 50 | `1e-4` | 完整验证集向量 MSE |
| `snr30_lr1e-4` | 最高 SNR，当前为30 dB | 100 | `1e-4` | 30 dB 验证向量 MSE |

诊断默认 batch 64、随机种子20261030，保留原 Adam 动量；第1轮、每5轮和末轮验证，原检查点始终可作为第0轮候选。预算由 `--joint-epochs`、`--snr-epochs` 和 `--batch-size` 控制，与正式200 epoch、batch 8的训练配置分开。30 dB 分支是从联合 SNR 模型继续训练的单点诊断，不能称为每个 SNR 分别训练的完整模型组。

```bash
python -m scripts.evaluate.oamp_net2_diagnostics \
  --data data/processed/nof1_compact_20261026_final \
  --checkpoint outputs/runs/oamp_net2_compensated/nof1/best.pt \
  --output outputs/runs/oamp_net2_diagnostics_20261030/nof1 \
  --device cuda --threads 4 --joint-epochs 50 --snr-epochs 100 --batch-size 64
```

NCS1 使用 `data/processed/ncs1_compact_20261026_nmse` 和对应 `ncs1/best.pt`，输出到同一诊断根目录的 `ncs1/`。每次诊断须选择新目录，该入口没有 `--resume`。输出为 `diagnostics.json`，以及三项试验各自的 `controls.json`、`history.json`、`result.json` 和标记 `diagnostic_only` 的 `best.pt`。这些权重不具备正式检查点格式，不可填入统一评测配置；诊断特征保留在该目录的 `features/`，完成后可按实验清理规则删除可重建缓存。

绘图入口 `scripts.visualize.oamp_net2_diagnostics` 从完成的报告读取数据，并要求根目录另有 `_control/validation_nmse.json`。该文件按信道保存 `snr_db` 和 `mean_nmse_db`，必须来自相同验证集的完整矩阵统计，不能用测试 NMSE 代替；诊断入口不自动生成它。两个信道均完成时，可在服务器项目根目录准备统计并画图：

```bash
python - <<'PY'
import json
from pathlib import Path
from src.data import CompactDataset
from src.evaluation.nmse import mean_nmse_db

config = json.loads(Path('configs/training/oamp_net2.json').read_text(encoding='utf-8'))
values = {}
for channel, directory in config['datasets'].items():
    with CompactDataset(Path(directory) / 'validation.h5', input_view='compensated') as dataset:
        values[channel] = {
            'snr_db': dataset.snr_db.tolist(),
            'mean_nmse_db': [mean_nmse_db(dataset, si) for si in range(len(dataset.snr_db))],
        }
control = Path('outputs/runs/oamp_net2_diagnostics_20261030/_control')
control.mkdir(parents=True, exist_ok=True)
(control / 'validation_nmse.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
PY
python -m scripts.visualize.oamp_net2_diagnostics --results outputs/runs/oamp_net2_diagnostics_20261030
```

可用 `--channel NOF1` 或 `--channel NCS1` 限定绘图。每信道生成验证集 `ber_snr.png`、共享输入 CSI 的 `nmse_snr.png` 和30 dB续训过程 `fine_tuning.png`。绘图只读取上述 JSON，不访问数据集、检查点或 GPU；30 dB 专项模型在 BER 图中只显示该点。诊断结果不构成新的测试集性能结论。

## 6. 验证

```bash
python -m pytest tests/algorithms/test_additional_equalizers.py tests/training/test_oamp_net2_training.py tests/training/test_oamp_net2_diagnostics.py -q
```

算法测试以独立的接收空间直接求逆为参考，比较 MMSE-PIC 多轮结果以及 OAMP-Net2
每层输出，覆盖有色噪声、矩形系统、四组标量梯度、秩亏、零信道、接收增益不变性，
并在 CPU 上核对 PyTorch 与 NumPy 路径。
集成测试覆盖固定划分、训练不访问测试、精确续训、raw/小样本检查点拒绝、标签访问拦截、
特征损坏重建、两种 MMSE 缓存保持、PIC 参数身份、完整六算法计数与离线重画。
这些小规模 CPU 测试验证实现与链路，不提供正式数据集上的性能结论。
