# 均衡算法增量评测

评测始终输出 Single-tap MMSE、Full-matrix LMMSE、EP、Robust EP 四条 BER 曲线，以及测试集平均 CSI NMSE 曲线。

## 运行

在项目根目录运行。服务器正式计算使用 CUDA，本地只运行 CPU 小数据测试。

```sh
python -m scripts.evaluate.compare --device cuda
python -m scripts.evaluate.compare --device cuda --force ep
python -m scripts.evaluate.compare --force single_tap_mmse robust_ep
python -m scripts.evaluate.compare --force all
python -m scripts.evaluate.compare --plot-only
```

`--force` 只强制所列算法重新计算，其余算法仍检查并复用有效缓存，最终报告保持四算法完整对比。`--channel NCS1` 或 `--channel NOF1` 可限制信道。

`--plot-only` 从配置输出目录下各信道的 `metrics.json` 重画两张图，不读取数据文件、不校验大文件、不初始化 GPU，并保留原始生成来源。它不能与 `--force` 一起使用。缺少 NMSE 数值的历史报告需先运行新版评测一次。

## 修改和独立测试

| 算法 | 实现 | 对应测试 |
| --- | --- | --- |
| Single-tap MMSE | `src/algorithms/MMSE/single_tap.py` | `tests/algorithms/test_single_tap_mmse.py` |
| Full-matrix LMMSE | `src/algorithms/MMSE/full_matrix.py` | `tests/algorithms/test_full_matrix_lmmse.py` |
| EP | `src/algorithms/ep/real.py` | `tests/algorithms/test_paper_ep.py` |
| Robust EP | `src/algorithms/robust_ep/real.py` | `tests/algorithms/test_robust_ep.py` |

EP 与 Robust EP 的迭代核心分别维护。批次算法接口是 `equalize_batch(batch, backend, ...)`，返回符号估计和逐帧拒绝计数。输入没有标签或参考 CSI 审计字段；Single-tap 使用原始观测，矩阵算法使用导频抵消观测。共享似然数组只读使用。

`configs/evaluation/ber_comparison.json` 的 `algorithms` 按四个稳定标识分别配置参数。旧顶层 `ep` 配置及 `evaluate()` 的旧参数作为两种 EP 的默认值，各算法显式配置优先。

## 缓存与来源

缓存存放在 `outputs/runs/_cache/`，与输出实验目录独立；测试或专用调用可通过 `evaluate(..., cache_dir=...)` 隔离缓存。数据和缓存不提交 Git。

每次计算前重新流式读取所需数据文件，计算 SHA256。相同内容移动目录仍可复用。缓存身份包含本算法代码、登记的计算依赖、实际参数、数据哈希、数值依赖和 BLAS/CUDA 环境；批大小、线程数、整个 Git 提交号和输出路径不参与失效。注释、文档字符串、格式和绘图样式不使均衡计算失效。

计算依赖登记在 `src/evaluation/registry.py` 与 `src/evaluation/fingerprints.py`。新增本地计算辅助函数时，必须将其依赖加入对应登记；共享辅助函数变化仅影响实际使用它的算法。输入准备或 BER 定义变化影响所有相关算法。

Robust EP 的训练协方差单独缓存，绑定训练数据、共享模型、CSI 缩放、SNR 及校准实现，不绑定 EP 迭代参数。训练数据单独变化时，普通算法结果仍可复用。

每算法按 SNR、帧和 D 块保存完成标记、内容校验和及生成会话。损坏块重算，中断从有效块续跑；操作系统锁阻止同键并发写入。强制重算失败时保留已有完整缓存，下一次正常运行优先恢复尚未完成的新尝试。运行中改写数据会中止并丢弃受影响的新缓存。

`metrics.json` 保存每条曲线的计算身份、实际生成提交、时间、环境及复用记录。全部算法成功且两图绘制完成后才更新正式产物，发布异常时回滚旧文件。旧格式检查点保留，但不自动导入新缓存。

NMSE 为每个测试帧、每个 D 的完整 CSI Frobenius 误差比，在线性域平均后转 dB。优先复用数据哈希匹配的验收统计，否则从测试数据计算；它作为数据集公共指标独立缓存，不随均衡算法修改重复计算。
