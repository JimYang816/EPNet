# 当前 ICR 训练和推理性能口径

当前统一设计及运行入口见 [ICR说明](icr_mmse.md)，公式和冻结预算结果见 [算法设计](icr_mmse_algorithm_design.md)。历史batch扫描、缓存迁移和旧门控实验仅保留在结果目录及Git完整备份中，不再作为当前默认配置。

两信道均8层、完整物理信道作用、修正半带宽16、窗口32/中心16/隐藏48，均学习全部物理参数和局部网络。训练为 `complex64`、`dense_device`、batch 8、SNR交错、预取2批、8192 MiB有界RAM缓存；不启用AMP或缩减物理基。`dense_device`在设备上准备完整矩阵后转为模型精度，正式输入不投影回DPSS系数空间。

参考矩阵缓存按需建立、受主存上限约束，不写入数据或检查点。它只供独立监督与指标计算。续训仍检查数据、控制参数、优化器和源码身份。

```bash
python -m scripts.evaluate.icr_training_perf --channel NOF1 --device cuda \
  --output outputs/runs/icr_training_performance_current/nof1 \
  --frames 128 --validation-frames 4 --repeats 3 --threads 4 \
  --reference-cache-mib 4096 --batch-sizes 2 4 8 16
```

入口从唯一配置解析对应信道的模型；测量后端、预取和缓存由该CLI参数决定。batch选择只读train/validation，不使用test选型。历史8层旧模型的吞吐量不可当作当前残差展开训练的实测速度。

当前冻结推理预算采用RTX4080、complex64、batch2、4个CPU线程，计入输入重建、全部展开层及CSI修正；一次性加载和冻结准备不计入稳态逐块预算。另以11次轮换模型计时的中位数复核，不以训练吞吐量替代推理延迟。

| 设置 | 原→当前总耗时 ms/块 | 当前/原预算 |
|---|---:|---:|
| NCS1固定CSI | 6.837→6.618 | 96.79% |
| NOF1固定CSI | 11.707→6.706 | 57.29% |
| NCS1迭代修正 | 9.091→8.875 | 97.63% |
| NOF1迭代修正 | 17.909→8.885 | 49.61% |

当前NOF1与NCS1均执行局部神经网络。NCS1预算余量较小，其他设备或配置应重新测量；固定CSI的单次完整矩阵LMMSE仍快于8层展开。预算基准是原Unfolding。NOF1原22层、新8层，不能把该表当作等深度的正式A–D消融。

完整效率入口 `scripts.evaluate.icr_efficiency` 接受明确的数据及学习检查点，每算法独立进程测量，并记录初始化、稳态、参数量和峰值内存。性能实验至少输出BER–SNR及平均NMSE–SNR，原始报告和不满足预期的指标均保留。
