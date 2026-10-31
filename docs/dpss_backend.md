# DPSS 信道算子后端与稠密对照

此实现保留当前 ICR-MMSE-Net 的参数、完整信道耦合、带状 CSI 修正、门控、阻尼、软反馈、归一化、噪声模式与三阶段损失。模型推理的信道算子可选择 `dense` 或 `dpss`，不改变网络结构，不重新拟合 DPSS，不截断基函数频谱。评测入口默认 `dense`；统一训练配置采用 `dense_device` 准备完整稠密输入。

统一配置 `configs/training/icr_mmse.json` 的 `training.operator_backend="dense_device"` 是稠密训练数据准备路径：双精度重建移到设备，随后转为配置精度的完整稠密矩阵；均衡时保留稠密算子。训练入口支持 `dense`、`dense_device`、`dpss`，独立评测入口支持 `dense`、`dpss`。训练配置中的 batch 8、预取2批与12288 MiB主存参考缓存的适用范围见 [训练性能说明](icr_mmse_training_performance.md)；配置及运行命令见 [ICR-MMSE-Net](icr_mmse.md#6-验证与运行)。

## 表示与等价边界

从已验收数据的 `samples/raw_coefficients` 和 `model.h5` 中的固定基读取输入，沿用 `DPSSModel.sampled_basis` 的 FFT 提前量与 D1/D2 物理窗口。保留全部 Q=34、L=736，N=1024。原始信道为 `H0`，当前迭代信道仍是 `Ht = H0 + Delta_t`，仅 `Delta_t` 使用原来的非循环物理带状支撑。

令 `h[n,l] = sum_q v_q[n] c_q[l]`，`Gq = FFT(cq, n=N)` 使用普通 FFT，F 为正交归一化 FFT。暂忽略边界的 fftshift 排序：

```
H0 x   = F sum_q [vq * F^H(Gq * x)]
H0^H r = sum_q [conj(Gq) * F(conj(vq) * F^H r)]
```

实现公共接口的向量与矩阵元素全部采用项目既有 fftshift 顺序；基底卷积为既有 `taps_to_matrix()` 的循环索引，修正支撑仍禁止跨 FFT 两端绕回。`DPSSChannel.dense()` 仅供独立诊断，正式 DPSS 推理路径不调用它。

`BandChannel` 将这两个原始算子与原有带状增量相加，完整基底的带外元素继续参与每次乘法。`H0[k,j] = sum_q FFT(vq)[(k-j) mod N] Gq[j] / N` 可直接计算全部反馈所需的局部元素。

## 精确辅助量

- 功率尺度使用 `||H0||F² = sum_l c_l^H(V^H V)c_l`；截取后的时间基 Gram 不假设为单位阵。
- 加权列能量保留每个频点的原噪声权重。基底部分通过基函数两两乘积与权重的循环相关核计算，不以平均噪声或带内能量代替。核按 D1/D2 基和精确权重共享，样本列能量按每次 forward 重新计算，轮内复用；增量部分继续使用原来的精确交叉项。
- 完整 CSI NMSE 的基底误差通过逆傅里叶变换参考矩阵与系数 Gram 恒等式计算，能量差以 float64 累加，再回到所选精度；不将参考矩阵投影截断后当作真值。修正交叉项和带内平方项保持完整矩阵误差定义。
- `full` 与 `diagonal` 两种噪声模式继续使用各自原有的协方差作用。DPSS 后端不改变噪声模型。

系数频谱、共享时间基、相关核与局部元素均参与正确的自动微分。权重和时间基要求梯度时不复用脱离计算图的相关核。输入视图及监督边界保持 raw 协议，参考 CSI 和标签不进入模型 forward。

## 使用

```
# DPSS 实验后端：训练/评测入口均支持。运行身份记录实际训练后端。
python -m scripts.train.icr_mmse --channel NOF1 --device cuda --operator-backend dpss
python -m scripts.evaluate.icr_mmse --channel NOF1 --checkpoint <checkpoint> --run outputs/runs/<evaluation> --device cuda --operator-backend dpss

# 先在独立 validation 集做 A/B/C 数值及参数梯度核对，失败即停止。
# 随后在 test 集的均匀抽样帧上做同检查点、全部 SNR、D1/D2 配对比较。
python -m scripts.evaluate.icr_operator --dataset <dataset>/test.h5 --checkpoint <checkpoint> --run outputs/runs/<comparison> --device cuda --max-frames 16 --timing-repeats 3

# 全部测试帧：
python -m scripts.evaluate.icr_operator --dataset <dataset>/test.h5 --checkpoint <checkpoint> --run outputs/runs/<full_comparison> --device cuda --full-test
```

同一检查点严格加载后同时用于两后端，不分别训练、调整噪声模式或修改精度。检查点仍要求当前源码身份；旧结构（例如原 564 通道修正网络）不能直接冒充当前 467 通道权重。`dense`、`dense_device` 或 `dpss` 新训练的检查点均可在稠密与 DPSS 算子间进行推理比较；严格续训还要求原训练后端、batch、预取、参考缓存及代码等训练身份一致。

`equivalence.json` 记录最低/最高 SNR 的一个验证帧、两 D 块、A/B/C 各阶段的算子、逐层概率、CSI 增量、完整 NMSE 和参数梯度误差。CPU 单元测试另覆盖奇偶 FFT、非正交复时间基、任意完整参考 CSI、系数/输入/噪声权重/时间基梯度和双精度。

`comparison.json`、`ber_snr.png`、`nmse_snr.png`、`runtime_memory.png` 保存配对结果。BER 采用已有软均值符号实虚部符号判决；NMSE 先平均完整矩阵线性相对平方误差，再转换 dB。零错误绘图使用 0.5/总比特数并说明。

耗时和显存在两个独立新进程测量，禁止读取监督字段。分别记录各后端的输入准备、重复前向中位数、输入准备加前向的耗时估计、首批冷前向、模型/数据打开时间、CUDA 已分配/保留峰值与 OS 进程 RSS 峰值。DPSS 每次前向清理样本派生缓存，避免重复计时省略首次列能量计算；跨样本共享基函数相关核与固定噪声分解可预热。显存峰值包含相关核建立及其他冷启动暂存，监督/评分不计入。

GPU 稠密核与 FFT 的舍入顺序不同，数学等价不要求逐比特浮点一致。完整学习性能结论需要正式训练及测试；短训和有限帧比较只验证替换与效率，不认定均衡性能达标。DPSS 的单次前向与端到端速度必须分别报告，Q=34 的常数和共享核内存可能限制收益。

真实数据高 SNR 下原生 FP32 的舍入差异可被八轮展开放大。默认评测仍要求原生概率/梯度等误差小于原严格阈值 5e-4（FP64 为 1e-9），失败即停止，不自动放宽。需要研究该差异的质量/效率影响时，显式添加 `--diagnose-native-drift`：保留 `equivalence.json` 的原生失败，另将同一检查点权重无损升为 complex128 验证两个后端，双精度控制不通过仍停止；只有该控制通过才测量原生精度的 BER/NMSE/耗时/显存。报告标记 `double_control_passed_native_strict_failed_diagnostic_only`，不将其当作原生严格等价验收，默认后端继续 dense。该诊断不修改模型权重、研究参数、数据、默认 TF32 设置或生产迭代精度。

## 验证

```
python -m pytest tests/algorithms/test_dpss_backend.py tests/evaluation/test_icr_operator.py tests/algorithms/test_icr_mmse.py tests/data/test_raw_csi.py tests/training/test_icr_mmse_training.py
```

产物只保存在 outputs/runs，Git 仅同步源码、测试与本文档。
