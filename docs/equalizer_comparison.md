# NOF1 / NCS1 BER 对比

入口为 `python -m scripts.evaluate.compare --device cuda --threads 2`，配置位于
`configs/evaluation/ber_comparison.json`。NOF1 使用 `nof1_compact_20261026_final`，
NCS1 使用最新调整后的 `ncs1_compact_20261026_nmse`。每信道输出 BER–SNR 图、
测试集平均 CSI NMSE–SNR 图及包含计算身份、真实生成来源和复用记录的指标。
增量缓存、强制重算和离线绘图的用法见 [增量评测说明](incremental_evaluation.md)。

沿用用户确认的 compact-v2 默认参考辅助构造 CSI，不重新补偿或改变测试数据。
这是既有构造数据条件下的算法对比，不代表原始接收机估计 CSI 的性能。
每信道使用全部 600 个测试帧，每 SNR 1200 个 D 块、1228800 比特；SNR 为 0:5:30 dB。
误码率为 D1/D2 所有比特总误码数除以总测试比特数。

- 单抽头 MMSE：原始数据载波观测、CSI 对角项和逐载波热噪声方差，忽略 ICI。
- 完整矩阵 LMMSE：所有接收行、所有数据列 ICI、估计 CSI 的已知导频消除，使用完整热噪声协方差。
- 传统 EP：同样的完整矩阵与导频消除，使用热噪声协方差。
- Robust EP：同样的完整矩阵与导频消除，使用热噪声加训练估计误差的完整有色协方差。

EP 和 Robust EP 均采用 Ghavami 与 Naraghi-Pour，2017，
DOI `10.1109/TVT.2017.2683527` 中的实数域公式 (23)–(34)。两种 EP 的核心实现和
参数分别维护，默认均为 10 次迭代、β=0.2、ε=5e-7，初始站点均值为 0、
方差为 Es/2=0.5；不合法更新保持原值。复协方差转换保留虚部对应的交叉块。
当前默认参数下，两种实现的检测方程一致，似然协方差不同；后续可分别修改与测试。

论文的解析 MIMO 信道误差模型不能直接代表本项目的 WATERMARK/DPSS 模型，
因此协方差采用独立训练集的残差二阶矩估计，并在测试前冻结。
对每 SNR、D 块，先计算 `e_raw=H_ref*x-H_raw*x` 与已保存的物理模型残差 `r_model`；
用现有测试集的公共 alpha 将训练残差转换为 `r_model+alpha*e_raw`。
训练信号包含实际训练数据及公开导频，因此统计同时包含数据 CSI 误差、导频消除误差和物理模型残差。
通过直接 DPSS 作用计算 `H_raw*x`，与完整频域矩阵精确等价，不截断带宽。

每训练帧用 `sqrt(rf_noise_variance*10**(snr_db/10))` 归一化该残差，
从全部训练帧估计复二阶矩。为避免 525 个训练样本估计 1024 维协方差时产生退化，
按训练样本的协方差方差自动估计向对角矩阵收缩的比例，保持各接收频点功率。
Robust EP 的单位 RF 协方差为 `R_thermal_unit+10**(snr_db/10)*R_error_normalized`，
再乘当前测试帧的 RF 热噪声方差。测试集真实误差、标签及 BER 均不参与此校准或参数选择。
已有系数协方差接口是接收机模型代理，本实现不把该代理直接当作真实误差协方差。

服务器结果位于 `outputs/runs/equalizer_comparison_20261026/{nof1,ncs1}/`。
`metrics.json` 保存误码数、总比特数、各算法参数、数据哈希、真实生成版本和缓存来源；
`ber_snr.png` 与 `nmse_snr.png` 保存两类指标图。训练协方差与各算法的逐块结果
分别缓存在 `outputs/runs/_cache/`，按内容身份跨实验复用，按 SNR/帧/D 块续跑。
旧格式 `calibration.h5` 和 `checkpoint.h5` 不自动导入新版缓存。
零误码以空心倒三角显示在 0.5/总比特数处，并明确标注，数值 BER 仍为零。
