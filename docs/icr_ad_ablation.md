# CSI 修正与 MMSE 展开的 A–D 消融

本轮在 NOF1、NCS1 已验收的 compensated CSI 和原有划分上运行全部 SNR。
按用户要求先使用一个配对种子 `20261028`，仅检验 A–D 四组，不开展门控、带宽或层数搜索。
单种子结果不能用于跨种子稳定性结论。配置为 `configs/training/icr_ad_ablation.json`。

| 组别 | CSI | 检测器 | 均衡器训练 | 修正器训练 |
| --- | --- | --- | --- | --- |
| A | 固定输入 CSI | 完整矩阵 LMMSE | 无 | 无 |
| B | 固定输入 CSI | MMSE 展开 | 预热 10 + 微调 10 epoch | 无 |
| C | 逐轮修正 | 完整矩阵 LMMSE | 无 | 预训练 10 + 微调 10 epoch |
| D | 逐轮修正 | MMSE 展开 | 预热 10 + 微调 10 epoch | 预训练 10 + 微调 10 epoch |

B、D 共用一份独立预热得到的阶段 A 均衡器权重。C、D 修正器及阻尼初始权重相同，修正器预训练阶段
只用 CSI NMSE，微调阶段用检测交叉熵与 CSI NMSE。修正器根据各自检测器的实际软反馈重新训练。
固定 CSI 的 B 跳过修正器预训练，继续完整的 10 轮检测微调，不能将单独的预热模型作为最终 B。
A 只有一次有效 LMMSE 求解；重复求解不更新固定 CSI 或单位能量先验，故不制造额外计算。

## 固定条件与传统检测器

NOF1 保持 22 层、`lambda_h=0.01`；NCS1 保持 8 层、`lambda_h=1`。B/C/D 在同一信道内具有相同层数，
修正半带宽均为 16。四组统一采用 complex64、完整基础信道矩阵、对角噪声权重、相同导频消除和物理网格。
沿用门控及显式方差反馈。A/C 的检测器本身没有可训练参数，也不使用上一轮软均值作为 LMMSE 先验；
反馈只能通过更新后的 CSI 影响下一次求解。对角噪声权重不等于对角信道，A/C 仍保留所有信道耦合。
本轮 A 不能与使用其他噪声模式的历史 Full-matrix LMMSE 缓存直接混用。

令 `W=diag(C)^-1`、`G=H_D^H W H_D`，每次求解

`x_hat=(G+sigma_RF^2 I)^-1 H_D^H W (y-H x_known)`。

使用 Cholesky 求解，并由高斯先验 LMMSE 的协方差
`V=sigma_RF^2 (G+sigma_RF^2 I)^-1` 构造固定的标量高斯近似软解映射。
令 `g_j=1-V_jj`，QPSK logits 为
`(2 Re(x_hat_j conj(a)) - g_j |a|^2)/V_jj`。
它等价于对去偏观测 `x_hat_j/g_j` 和有效方差 `V_jj/g_j` 使用星座距离，消去了与类别无关的项，
避免在零增益时直接除以零。该软映射是近似分布，不宣称为不完美 CSI 下的真实后验。
最后通过概率计算均值和方差；对 QPSK 的比特硬判决与 LMMSE 估计实虚部符号一致。
求解及软映射对当前 CSI 保留梯度，使 C 组微调时检测损失能训练修正器。

## 训练与验收

训练 batch 8，交错覆盖 SNR，每个物理帧及 D 块每轮出现一次。A/B 训练阶段学习率 1e-3；
微调时均衡器学习率 1e-4、修正器及阻尼学习率 1e-5。C/D 的预训练和微调采样序列一致，B/D 的微调序列一致。
固定参数阶段使用零 epoch，保存阶段边界及审计信息，不执行虚假的优化更新。

沿用逐 SNR 零退化验收：修正器预训练相对于其起点、微调相对于其起点，BER 和完整矩阵 NMSE 均不能恶化。
起点始终作为候选，无合格改进时回退并报告选中 epoch=0。零退化限制可能导致候选回退；该行为属于预先确定的
训练协议，不能因为测试结果不理想而放宽。四组全部完成验证选模以后才开启测试。

`training_audit.json` 校验共同预热权重、修正器初始化、采样顺序和有效训练轮数。
每个 epoch 保存优化器、随机状态和所选权重；中断后从已完成 epoch 恢复，未完成的 epoch 重做。
数据、代码、精度或控制参数变化时拒绝续训，不复用 raw CSI 或历史不同代码的检查点。

## 运行与产物

```bash
.venv/bin/python -u -m scripts.run.icr_ad_ablation --device cuda --gpu 0 --threads 4 --resume
# 隔离代码工作区可以共享项目中的数据和输出：
/home/fay/JimYang/EPNet/.venv/bin/python -u -m scripts.run.icr_ad_ablation \
  --project-root /home/fay/JimYang/EPNet --device cuda --gpu 0 --threads 4 --resume
```

正式运行放在服务器独立 systemd 用户服务中；系统启用 linger，因此本地电脑或 SSH 断开不终止实验。
单 GPU 文件锁阻止本项目其他遵循相同锁的任务并发占用。后台服务保存日志，失败状态和 traceback 单独留存，
服务以有限重试恢复；成功完成退出，不循环重跑。服务器服务配置放在输出目录 `_control/`，不提交到 Git。
启动时应先做完整尺寸 GPU 前向/反向资源检查，再开始全量训练。

输出目录为 `outputs/runs/icr_ad_ablation_20261030/`。两信道串行运行，各组保存检查点、训练历史、逐 SNR
和逐物理记录测试指标、逐层 BER/NMSE。汇总生成 `comparison.json/csv`、`seed_summary.json`、
`ber_snr.png/svg` 和 `nmse_snr.png/svg`。每完成一个信道更新汇总，单信道图不代表整个任务完成。
配对 BER 差为 B−A、D−C、C−A、D−B，负值代表前者改善；以整条物理记录进行 bootstrap。
NMSE 先平均完整矩阵逐块线性值，最后转 dB。时延包括输入准备、预处理及全部有效迭代，排除监督读取与评分；
本轮不宣称计算预算匹配。报告有效推理参数数，另存模型容器内包含未使用分支的参数数。
CPU 峰值 RSS 为进程生命周期峰值，不把同一进程内的各组 RSS 当成独立的内存比较。
完成一信道后删除冗余 `latest.pt` 恢复副本，保留正式权重、训练与验收记录。
