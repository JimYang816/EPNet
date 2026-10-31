# 独立训练与增量评测

当前七算法正式结果已保存在 `outputs/runs/equalizer_comparison_mmse_dfe_20261031/{nof1,ncs1}/`，包含双信道完整BER/NMSE图及报告，根目录 `verification.json` 记录新增MMSE-DFE后原六条曲线数值保持一致的核验。原六算法完整训练与评测结果仍保存在 `outputs/runs/icr_full_20261031_095549/`。
服务器保留 ICR 的 warmup/fixed/joint 检查点、UDNet 的 best/latest 检查点、原 OAMP-Net2 检查点及逐算法结果缓存。
本地保存图、指标、训练记录与报告；完整数据、特征缓存及权重仍留服务器。

后续只评测或更新对比图时使用下列入口，它不调用任何训练函数：

```bash
python -m scripts.evaluate.compare --device cuda --threads 4
python -m scripts.evaluate.compare --device cuda --threads 4 --channel NCS1
python -m scripts.evaluate.compare --plot-only
```

默认配置为 `configs/evaluation/ber_comparison.json`，ICR引用 `outputs/runs/icr_full_20261031_095549/joint/<channel>/icr_mmse.pt`，UDNet引用同一实验的 `udnet/<channel>/best.pt`，OAMP-Net2引用 `outputs/runs/oamp_net2_compensated/<channel>/best.pt`；其中 `<channel>` 为 `nof1` 或 `ncs1`。
改变某个学习算法时，单独训练该算法到新的实验目录，将对应信道、算法的 checkpoint 路径切到新权重，再执行统一评测。
缺少或不匹配的检查点会报错，不自动训练其他模型。`scripts.run.icr_experiment` 是显式的完整训练入口，不用于日常增量评测。

复用按实际计算身份判断：数据内容、compensated 视图、算法参数、对应权重、算法依赖源码和数值环境。
独立修改 UDNet 不使 MMSE/DFE/PIC/OAMP/ICR 的结果失效；独立修改 ICR 也不重算其他检测器。
修改公共数据构造或评分逻辑则会使实际受影响的算法重算。文档、输出目录和独立绘图脚本的修改不改变数值结果身份。

传统方法、UDNet、OAMP-Net2 使用已有逐块校验缓存。ICR 完整报告缓存位于 `outputs/runs/_cache/icr_reports/`，
同时核验数据、权重摘要、原推理/评分代码、bootstrap 设置和每 SNR 全量比特覆盖。
首次可直接导入当前检查点所属实验中身份一致的历史报告，无需重放推理；缓存本身保存 SHA256 校验。
复用记录写入 `icr_mmse.evaluation_cache`；其中耗时保留原测量来源，不能作为新的效率实测。

只有显式需要重算时使用 `--force mmse_dfe`、`--force mmse_pic`、`--force icr_mmse`，也可用对应标识指定两种LMMSE；`--force` 支持同时传入多项，`--force all` 会刷新已配置的传统方法、OAMP-Net2 和 ICR。UDNet仍按自身计算身份复用缓存，统一CLI不提供单独强制重算UDNet的选项。
历史训练、报告和不同身份的缓存保留，不用重新训练全部算法来更新单条曲线。
