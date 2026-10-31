# MMSE-DFE

实现位于 `src/algorithms/mmse_dfe/detector.py`，标识为 `mmse_dfe`，图例为 **MMSE-DFE**。
这是固定 CSI 的完整矩阵块判决反馈均衡器，无需训练、没有可调检测参数。
检测顺序固定为数据列从末列到首列，不使用最优排序或测试标签选序。

## 计算定义

沿用统一评测的 compensated CSI、全部 FFT 接收行、全部数据列和估计导频消除。
对完整热噪声协方差 `sigma_rf² C` 白化后，共享似然量为：

```text
G = Hᴴ (sigma_rf² C)^(-1) H
g = Hᴴ (sigma_rf² C)^(-1) y
I + G = L Lᴴ,  R = Lᴴ
z = L^(-1) g

for i = N-1, ..., 0:
    u_i = (z_i - sum_{j>i} R_ij * xhat_j) / R_ii
    xhat_i = QPSK_slice(u_i)
```

单位阵对应单位平均能量 QPSK 的 MMSE 正则项。
首次检测与完整矩阵 LMMSE 的对应分量一致；之后仅反馈已检测符号的硬 QPSK 判决。
不把线性估计值直接当作反馈，否则会退化为完整矩阵 LMMSE。
输出给公共 BER 计分器的是判决前 `u`；实部或虚部恰为零时判为正，与项目比特映射一致。
固定顺序允许实际判决错误继续传播，不使用理想反馈。

该块 DFE 对应增广白化信道 `[H_white; I]` 的 QR 分解，使用 Cholesky 分解计算等价因子。
相关块 MMSE-DFE 分解方法见 Kusume、Joham、Utschick，
[*MMSE Block Decision-Feedback Equalizer for Spatial Multiplexing with Reduced Complexity*](https://mediatum.ub.tum.de/doc/620740/620740.pdf)，GLOBECOM 2004。
本实现固定数据列顺序，没有实现该文的最优检测排序。

支持 NumPy CPU 和 PyTorch CUDA，保持 `complex128/float64`。
独立测试使用接收空间逐步重建协方差、直接 MMSE 滤波并消除硬判决，核验有色噪声、非连续数据列、秩亏及零信道。

## 运行与指标

`configs/evaluation/ber_comparison.json` 已加入 `"mmse_dfe": {}`，默认统一比较七种算法。

```bash
python -m scripts.evaluate.compare --device cuda --threads 4
python -m scripts.evaluate.compare --device cuda --baselines-only
python -m scripts.evaluate.compare --channel NOF1 --device cuda --force mmse_dfe
python -m scripts.evaluate.compare --plot-only
```

正式测试仍为每信道、每 SNR 600 帧、D1/D2、512 个数据槽，共 1,228,800 比特。
新增结果单独缓存；已有其他算法及 NMSE 的有效缓存继续复用。
`ber_snr.png` 包含七条曲线；`nmse_snr.png` 中 MMSE-DFE 共用固定输入 CSI 的平均 NMSE，
ICR-MMSE-Net 另报修正后的平均 NMSE。检测器不修正 CSI，因此不另造一条重合的 NMSE 曲线。
默认输出为 `outputs/runs/equalizer_comparison_mmse_dfe_20261031/{nof1,ncs1}/`，保留既有六算法实验目录。

## 正式结果（2026-10-31）

已在 `lab448` 的 RTX 4080 完成双信道正式测试，计算代码提交为 `b93e836`。
使用 `--device cuda --threads 4 --batch-size 32`，每信道覆盖 8,400 个 D 块，
每 SNR 为 1,228,800 比特；CPU/CUDA 独立数值测试全部通过（8 项）。

| 信道 / MMSE-DFE BER | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| NOF1 | 0.17123210 | 0.07774821 | 0.03018392 | 0.01220296 | 0.00625244 | 0.00435140 | 0.00372477 |
| NCS1 | 0.21192301 | 0.11305420 | 0.05534098 | 0.03072347 | 0.02175130 | 0.01836100 | 0.01679362 |

30 dB 时相对 Full-matrix LMMSE 的实测 BER 分别下降 16.16% 和 19.46%。
原六算法的误码计数、BER、输入/修正 CSI 平均 NMSE 与原统一报告逐项一致，未重新训练模型。
两信道七曲线 BER 图和平均 NMSE 图已通过目视检查及 SHA256 校验，并通过 SSH 带回本地。
完整误码计数与来源见各信道 `metrics.json`，本地汇总核验记录为该实验根目录的 `verification.json`。
生成数据、数值结果、检查点和指标图均不提交到 Git。
