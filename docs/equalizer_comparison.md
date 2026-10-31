# NOF1 / NCS1 均衡评测与增量缓存

入口为 `python -m scripts.evaluate.compare --device cuda --threads 2`，配置位于
`configs/evaluation/ber_comparison.json`。NOF1 使用 `nof1_compact_20261026_final`，
NCS1 使用最新调整后的 `ncs1_compact_20261026_nmse`。每信道输出 BER–SNR 图、
测试集平均 CSI NMSE–SNR 图及包含计算身份、真实生成来源和复用记录的指标。
本页统一说明算法比较、增量缓存、强制重算与离线绘图。

主实验默认 `input_view="compensated"`，按已保存的 alpha 重建完整 `H_init=H_ref+alpha*(H_raw-H_ref)`。
用户固定平均 NMSE：NOF1 为 −12.70→−18.30 dB，NCS1 为 −6.20→−10.67 dB（SNR 0→30 dB）。
接收观测、数据文件和既定记录划分保持不变。参考 CSI 和 alpha 只用于已验收的数据构造；算法不额外接收参考 CSI 或标签。raw 视图仅供显式诊断，不是默认训练评测输入；联合修正方案见
[ICR-MMSE-Net](icr_mmse.md)。
每信道使用全部 600 个测试帧，每 SNR 1200 个 D 块、1228800 比特；SNR 为 0:5:30 dB。
误码率为 D1/D2 所有比特总误码数除以总测试比特数。

- 单抽头 MMSE：原始数据载波观测、CSI 对角项和逐载波热噪声方差，忽略 ICI。
- 完整矩阵 LMMSE：所有接收行、所有数据列 ICI、估计 CSI 的已知导频消除，使用完整热噪声协方差。

统一基线评测仅包含上述两种 MMSE。UDNet 独立评测在此基础上加入 UDNet，共三种算法；
ICR-MMSE-Net 完整流水线再加入 ICR-MMSE-Net，共四种算法。
EP、Robust EP 不再参与对比，也不再拟合或加载其训练误差协方差。
基线评测仅校验并读取 `test.h5` 和 `model.h5`，不依赖 `train.h5`。

默认结果目录为 `outputs/runs/equalizer_comparison_compensated/{nof1,ncs1}/`；
历史 `equalizer_comparison_20261026` 结果属于此前输入视图。
`metrics.json` 保存误码数、总比特数、各算法参数、数据哈希、真实生成版本和缓存来源；
`ber_snr.png` 与 `nmse_snr.png` 保存两类指标图。NMSE 与各算法的逐块结果
分别缓存在 `outputs/runs/_cache/`，身份包含输入视图，按内容身份跨实验复用，按 SNR/帧/D 块续跑。
旧格式 `calibration.h5` 和 `checkpoint.h5` 不自动导入新版缓存。
零误码以空心倒三角显示在 0.5/总比特数处，并明确标注，数值 BER 仍为零。

## 运行控制

```bash
python -m scripts.evaluate.compare --device cuda
python -m scripts.evaluate.compare --device cuda --channel NCS1 --force full_matrix_lmmse
python -m scripts.evaluate.compare --force single_tap_mmse full_matrix_lmmse
python -m scripts.evaluate.compare --force all
python -m scripts.evaluate.compare --plot-only
```

`--force` 仅重算所列算法，其余算法检查并复用有效缓存，报告始终包含两种 MMSE。`--channel NOF1` 或 `--channel NCS1` 限定信道。`--plot-only` 从配置输出目录的 `metrics.json` 重画BER与NMSE图，不读取数据、不初始化GPU，保留生成来源和输入视图；它不能与 `--force` 同用。重画历史图沿用原视图，并移除 EP、Robust EP 曲线及其专用报告字段。

## 算法维护与缓存身份

| 算法 | 实现 | 对应测试 |
|---|---|---|
| Single-tap MMSE | `src/algorithms/MMSE/single_tap.py` | `tests/algorithms/test_single_tap_mmse.py` |
| Full-matrix LMMSE | `src/algorithms/MMSE/full_matrix.py` | `tests/algorithms/test_full_matrix_lmmse.py` |

批次接口为 `equalize_batch(batch, backend, ...)`，返回符号与逐帧拒绝计数。输入不含标签或参考CSI，纯似然数组共享只读。配置 `algorithms` 仅接受 `single_tap_mmse`、`full_matrix_lmmse`，当前两者参数均为空对象。EP 标识及其迭代参数不再接受。

每次计算前流式校验所需数据文件SHA256。相同内容移动目录可复用缓存。计算身份绑定输入视图、算法和登记的依赖、实际参数、数据哈希、数值依赖及BLAS/CUDA环境；batch、线程数、Git提交号和输出路径不使缓存失效。注释、文档和绘图样式修改不影响数值缓存。新增计算辅助函数时，在 `src/evaluation/registry.py` 与 `src/evaluation/fingerprints.py` 登记实际依赖。

两种 MMSE 保留原有算法计算身份，可继续复用同视图、同数据和同数值环境下的有效缓存；移除其他算法不触发 MMSE 重算。NMSE按视图和数据身份缓存，按每块完整矩阵线性误差比平均后转dB；缺少视图标记的历史验收曲线按compensated处理。

结果缓存逐SNR/帧/D块保存完成标记、校验和及生成会话。损坏块重算，中断从有效块续跑；操作系统锁阻止同键并发写入。强制重算失败保留原完整缓存，下次优先恢复未完成的新尝试。运行中数据变化会中止并丢弃受影响的新缓存。测试可通过 `evaluate(..., cache_dir=...)` 隔离缓存。

`metrics.json` 记录实际生成提交、环境、计算身份与复用来源。两种 MMSE 和两图全部成功后发布正式产物，发布失败回滚旧文件；旧格式检查点不自动导入新版缓存。
