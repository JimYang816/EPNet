# DPSS 实验算子与完整矩阵后端

当前正式实验使用默认读取器 `input_view="compensated"` 返回的完整估计 CSI：训练采用 `dense_device`，评测采用 `dense`。本页另记录原始系数视图（`raw`）下的 DPSS 算子诊断；该诊断保留 ICR-MMSE-Net 的参数、完整信道耦合、带状修正、门控、阻尼、软反馈、归一化、噪声模式与三阶段损失，不重新拟合 DPSS 或截断基函数频谱。只有显式要求 raw 诊断时才使用该路径，其检查点、特征及评测缓存不用于正式六算法对比。

统一配置 `configs/training/icr_mmse.json` 的 `training.operator_backend="dense_device"` 是完整矩阵训练数据准备路径：双精度计算移到设备，随后转为模型精度；均衡时保留稠密算子。默认估计 CSI 不受原始 DPSS 系数子空间限制，单独的 `dpss` 算子不能表示其完整输入，因此正式训练应使用 `dense` 或 `dense_device`，正式评测使用 `dense`。解析器保留的 `dpss` 选项不代表默认输入支持该后端。当前batch 8、预取2批与8192 MiB主存参考缓存见 [训练性能说明](icr_mmse_training_performance.md)；运行命令见 [ICR-MMSE-Net](icr_mmse.md#6-验证与运行)。

## 诊断表示与等价边界

独立诊断显式选择 `input_view="raw"`，从 `samples/raw_coefficients` 和 `model.h5` 中的固定基读取输入，沿用 `DPSSModel.sampled_basis` 的 FFT 提前量与 D1/D2 物理窗口。保留全部 Q=34、L=736，N=1024。本节将系数重建信道记为 `H0`，当前迭代信道为 `Ht = H0 + Delta_t`，仅 `Delta_t` 使用非循环物理带状支撑。本节的系数恒等式仅适用于该诊断视图。

令 `h[n,l] = sum_q v_q[n] c_q[l]`，`Gq = FFT(cq, n=N)` 使用普通 FFT，F 为正交归一化 FFT。暂忽略边界的 fftshift 排序：

```
H0 x   = F sum_q [vq * F^H(Gq * x)]
H0^H r = sum_q [conj(Gq) * F(conj(vq) * F^H r)]
```

实现公共接口的向量与矩阵元素全部采用项目既有 fftshift 顺序；基底卷积为既有 `taps_to_matrix()` 的循环索引，修正支撑仍禁止跨 FFT 两端绕回。`DPSSChannel.dense()` 仅供显式矩阵核对，DPSS 算子推理分支不调用它。

`BandChannel` 将这两个系数基底算子与带状增量相加，完整基底的带外元素继续参与每次乘法。`H0[k,j] = sum_q FFT(vq)[(k-j) mod N] Gq[j] / N` 可直接计算全部反馈所需的局部元素。

## 精确辅助量

- 功率尺度使用 `||H0||F² = sum_l c_l^H(V^H V)c_l`；截取后的时间基 Gram 不假设为单位阵。
- 加权列能量保留每个频点的原噪声权重。基底部分通过基函数两两乘积与权重的循环相关核计算，不以平均噪声或带内能量代替。核按 D1/D2 基和精确权重共享，样本列能量按每次 forward 重新计算，轮内复用；增量部分继续使用原来的精确交叉项。
- 完整 CSI NMSE 的基底误差通过逆傅里叶变换参考矩阵与系数 Gram 恒等式计算，能量差以 float64 累加，再回到所选精度；不将参考矩阵投影截断后当作真值。修正交叉项和带内平方项保持完整矩阵误差定义。
- `full` 与 `diagonal` 两种噪声模式继续使用各自原有的协方差作用。DPSS 后端不改变噪声模型。

系数频谱、共享时间基、相关核与局部元素均参与正确的自动微分。权重和时间基要求梯度时不复用脱离计算图的相关核。该诊断固定使用 `raw` 系数视图，模型 forward 不接收参考 CSI 或标签。

## 正式运行与独立诊断

正式训练与评测使用完整矩阵路径：

```bash
python -m scripts.train.icr_mmse --channel NOF1 --device cuda --operator-backend dense_device
python -m scripts.evaluate.icr_mmse --channel NOF1 --checkpoint <checkpoint> --run outputs/runs/<evaluation> --device cuda --operator-backend dense
```

DPSS 对照通过独立的 `scripts.evaluate.icr_operator` 入口运行。该入口固定读取 `raw` 系数视图，要求同一视图且源码、结构和数据身份均匹配的诊断检查点；默认正式模型检查点不适用于此入口。

```bash
# 先在独立 validation 集做 A/B/C 数值及参数梯度核对，失败即停止。
# 随后在 test 集的均匀抽样帧上做同检查点、全部 SNR、D1/D2 配对比较。
python -m scripts.evaluate.icr_operator --dataset <dataset>/test.h5 --checkpoint <raw-checkpoint> --run outputs/runs/<comparison> --device cuda --max-frames 16 --timing-repeats 3

# 全部测试帧：
python -m scripts.evaluate.icr_operator --dataset <dataset>/test.h5 --checkpoint <raw-checkpoint> --run outputs/runs/<full_comparison> --device cuda --full-test
```

同一诊断检查点严格加载后同时用于两后端，不分别训练、调整噪声模式或修改精度。后端间比较要求输入视图一致，并保留完整的源码和数据校验；不能据此跨输入视图复用检查点。旧结构（例如原 564 通道修正网络）不能直接冒充当前 467 通道权重。严格续训还要求原训练后端、batch、预取、参考缓存及代码等训练身份一致。

`equivalence.json` 记录最低/最高 SNR 的一个验证帧、两 D 块、A/B/C 各阶段的算子、逐层概率、CSI 增量、完整 NMSE 和参数梯度误差。CPU 单元测试另覆盖奇偶 FFT、非正交复时间基、任意完整参考 CSI、系数/输入/噪声权重/时间基梯度和双精度。

`comparison.json`、`ber_snr.png`、`nmse_snr.png`、`runtime_memory.png` 保存配对结果。BER 采用已有软均值符号实虚部符号判决；NMSE 先平均完整矩阵线性相对平方误差，再转换 dB。零错误绘图使用 0.5/总比特数并说明。

耗时和显存在两个独立新进程测量，禁止读取监督字段。分别记录各后端的输入准备、重复前向中位数、输入准备加前向的耗时估计、首批冷前向、模型/数据打开时间、CUDA 已分配/保留峰值与 OS 进程 RSS 峰值。DPSS 每次前向清理样本派生缓存，避免重复计时省略首次列能量计算；跨样本共享基函数相关核与固定噪声分解可预热。显存峰值包含相关核建立及其他冷启动暂存，监督/评分不计入。

GPU 稠密核与 FFT 的舍入顺序不同，数学等价不要求逐比特浮点一致。完整学习性能结论需要正式训练及测试；短训和有限帧比较只验证替换与效率，不认定均衡性能达标。DPSS 的单次前向与端到端速度必须分别报告，Q=34 的常数和共享核内存可能限制收益。

真实数据高 SNR 下原生 FP32 的舍入差异可被八轮展开放大。独立诊断默认要求原生概率/梯度等误差小于原严格阈值 5e-4（FP64 为 1e-9），失败即停止，不自动放宽。需要研究该差异的质量/效率影响时，显式添加 `--diagnose-native-drift`：保留 `equivalence.json` 的原生失败，另将同一检查点权重无损升为 complex128 验证两个后端，双精度控制不通过仍停止；只有该控制通过才测量原生精度的 BER/NMSE/耗时/显存。报告标记 `double_control_passed_native_strict_failed_diagnostic_only`，不将其当作原生严格等价验收，默认后端继续 dense。该诊断不修改模型权重、研究参数、数据、默认 TF32 设置或生产迭代精度。

## 验证

```
python -m pytest tests/algorithms/test_dpss_backend.py tests/evaluation/test_icr_operator.py tests/algorithms/test_icr_mmse.py tests/data/test_raw_csi.py tests/training/test_icr_mmse_training.py
```

产物只保存在 outputs/runs，Git 仅同步源码、测试与本文档。
