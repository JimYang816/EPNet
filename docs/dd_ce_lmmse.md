# 判决导向信道估计与完整矩阵 LMMSE

主对比中的 `dd_ce_lmmse` 使用相同的 Full-matrix LMMSE 做两次检测，中间用第一次检测的硬 QPSK 符号更新一次 CSI。与固定 CSI 的 Full-matrix LMMSE 配对，可直接比较这次更新对 BER 和信道估计精度的影响。水声 OFDM 中将检测数据作为虚拟导频的背景见 [Radosevic 等，Allerton 2011](https://doi.org/10.1109/Allerton.2011.6120229)。下述降秩估计器为本项目的实现。

1. 以已验收的完整 compensated CSI 矩阵 $H_0$ 消除已知导频，用全部接收行、数据列和完整热噪声协方差求 LMMSE。将输出切片到最近的单位能量 QPSK 星座点，和已知导频、零载波拼成完整向量 $x$。
2. 将接收机已有的 DPSS 基截取到当前 D 块，按它的独立延迟块先验构造时域抽头协方差。取最大 32 个特征值对应的时变抽头模式，模式包含特征值平方根。每个模式转成完整频域矩阵 $B_i$，其对当前已判符号的响应组成 $A=[B_1 x,\ldots,B_r x]$。D1、D2 各有自己的模式。先验来源为数据集原有的独立记录，两信道使用同一选模规则。
3. 记 $W C W^H=I$，其中接收热噪声协方差为 $\sigma^2 C$。用复数岭回归估计误差系数，再重建信道

   $$
   \hat a=\arg\min_a\ \|W(y-H_0x-Aa)\|_2^2+(\sigma^2/\eta)\|a\|_2^2,
   \qquad H_1=H_0+\sum_i \hat a_i B_i .
   $$

   在两个信道、全部 SNR 上固定 $\eta=0.03$。秩、先验方差比例和一次更新的设置在正式测试前确定。降秩约束只作用于增量，初始 CSI 的完整矩阵以及载波间干扰均参与检测。
4. 用 $H_1$ 重新消除导频并重新计算白化似然，运行相同的 Full-matrix LMMSE。每个数据块独立处理。更新器接收含噪观测、初始 CSI、公开导频、接收机噪声和先验，不读取真实数据符号或参考信道。

## 评测

`configs/evaluation/ber_comparison.json` 同时启用固定 CSI 与 DD-CE 两条 LMMSE 曲线。可单独重算新增基线。

```bash
python -m scripts.evaluate.compare --device cuda --threads 4 --force dd_ce_lmmse
```

`nmse.dd_mean_nmse_db` 对每帧每 D 块的 $\|H_1-H_{\rm ref}\|_F^2/\|H_{\rm ref}\|_F^2$ 先做线性平均，再取 $10\log_{10}$。该指标与最终 BER 使用同批帧，逐块缓存同时保存误码数和线性 NMSE。
