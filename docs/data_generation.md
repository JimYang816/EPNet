# NOF1 / NCS1 统一数据生成

数据生成已完成，当前训练与评测复用已验收数据及既定划分。本页记录数据协议、读取接口和生成流程；生成入口只执行数据生成、落盘验收和 Python 读取，不训练模型，不运行均衡器或 BER。两信道共用固定 QPSK P1–D1–D2–P2 链路：FFT1024、CP384、提前32点、每D512数据与128导频，6/96kHz采样、14kHz载频。DPSS仍为5632点、半带宽15.686274509803921Hz、尾比1e-5、Q34、L736、延迟块32。

## 划分与协议

当前协议为 `watermark-pddp-compact-v2`，旧版全CSI导出、读取与均衡流程已移除。两份 `configs/data/*_pddp.json` 除信道名外相同，沿用已验收版本。`learning_dataset` 是学习数据的唯一划分来源；根层 `prior_records`、`validation_records` 和 `starts_seconds` 用于CSI先验及估计器校验。旧 `test_records`、奇偶窗口和旧EP字段作为原始配置来源记录保留，当前入口不使用；保留这些字段维持数据配置身份。

- 先验：051–055；CSI估计器校验：056–060，窗口沿用2/8/14/20/26秒。后者不是学习验证集。
- 学习测试：011/015/019/023/027/031/035/039/043/047；学习验证：013/021/029/037/045。
- 学习训练：001–050中排除学习验证和测试的35条记录。
- 每条学习记录窗口为2:2:30秒；SNR为0:5:30dB。

| 集合（每信道） | 物理窗口 | 每窗重复 | 每SNR帧 | 全SNR D块 | 每SNR比特 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 525 | 1 | 525 | 7350 | 1075200 |
| validation | 75 | 1 | 75 | 1050 | 153600 |
| test | 150 | 4 | 600 | 8400 | 1228800 |

载荷与标准RF噪声各自使用独立编号和随机流：`SeedSequence([master, split, record, start_ms, stream, repeat])`，stream=0为载荷、1为噪声。当前测试将两个编号一一配对为0–3。每个重复跨SNR共用载荷和标准噪声；公开导频流固定为`[master,11]`。记录、所有窗口、D1/D2和重复都保持在同一集合。

## 估计 CSI 与不确定性代理

默认读取器的 `input_view="compensated"` 返回已验收的完整估计 CSI，记为 `H_est`。当前七算法使用同一固定输入；raw 视图仅供显式诊断，不用于主实验，也不得复用其检查点、特征或评测缓存。平均 NMSE 先对全部 D 块的线性相对平方误差求均值，再转换为 dB。

`m(s)=a-b*(1-exp(-s/10))/(1-exp(-3))`；NOF1为(a,b)=(-12.7,5.6)，NCS1为(-6.2,4.47)。NCS1在SNR=0/5/10/15/20/25/30dB的目标显示为−6.20/−8.05/−9.17/−9.85/−10.27/−10.52/−10.67dB；计算使用未取整公式值。

NOF1 平均 NMSE 随 SNR 0→30 dB 为 −12.70→−18.30 dB，NCS1 为 −6.20→−10.67 dB。这些目标及数据划分已经固定，不根据 BER 调整。数据文件保存物理观测、参考 CSI、标签及噪声参数的来源哈希，正式训练与评测启动时校验输入和目标的一致性。

协方差只称为**接收机模型代理**。共享文件保存块结构先验P及估计器已有的白化观测谱(lambda,G)，原始系数代理为：

`C_proxy = P - G @ diag(1/(lambda + effective_rf_variance + ridge)) @ G.H`。

提供流式协方差乘法、trace和按调用方指定数量提取模式的接口，不保存25024×25024矩阵。有效方差保留现有估计器基于公开P1/P2功率的CSI模型误差下限。

每 SNR 的代理标定统计仅从训练集拟合，验证与测试使用冻结统计。参考投影在同一整帧 DPSS 基上对四个 FFT 窗做最小二乘。读取器返回与估计 CSI 对应的协方差接口，调用方无需额外缩放；该代理不代表包含全部模型失配的真实误差协方差。

训练 CSI 投影残余的逐接收频点功率包含未知单位符号能量和确定公开导频贡献，读取时使用已保存的标定统计。物理残差独立保存为 `r_model=y_clean-H_ref@x`。两者分别提供，不合并成所谓真实误差协方差。

## 存储与读取

每信道、run目录结构：

```text
data/processed/<run-id>/
  model.h5          # DPSS、公开信号、掩码、星座、单位RF接收协方差、块先验、观测谱、训练统计
  train.h5
  validation.h5
  test.h5
  manifest.json     # 数量、目标、来源哈希、文件大小/哈希、验收结果
```

split文件按物理窗口保存一份完整D1/D2参考频域CSI（complex64，全部耦合），不做DPSS投影或带状截断。每样本、SNR保存原始整帧DPSS系数（complex64）和有效RF方差。参考投影、干净P1/D1/D2/P2观测及标准接收噪声以complex128保存；含噪观测准确重建为`clean+sqrt(rf_variance)*standard_noise`。标签保存比特，QPSK复符号、四分类索引及映射准确重建。类别为`2*b0+b1`，映射依次为00、01、10、11。

未压缩大小以清单为准；不逐样本重复保存大型估计矩阵。LZF按窗口或样本分块。每个split还保存记录、窗口、载荷/噪声编号、种子、功率/SNR诊断与物理模型残差。

```python
from src.data import CompactDataset, data_system

with CompactDataset('data/processed/nof1_compact_20261026_final/test.h5',
                    input_view='compensated') as dataset:
    inputs, labels = dataset[0]   # SNR-major；每个item是一帧，显式D1/D2轴
    # inputs: observations[4,N], csi[2,N,N], noise, known[4,N],
    #         masks, data_indices, pilot_indices, carrier_positions
    # labels: bits[2,512,2], symbols[2,512], classes[2,512], constellation及其比特映射
    y_data, H_data = data_system(inputs)
    # y_data = y_D - H_est @ known_D; H_data = H_est[..., data_indices]
    audit = dataset.audit(0, 0)           # 仅显式监督/核查时使用
```

默认读取器通过 `csi` 返回完整已验收的估计 CSI。`model_input(si,fi)` 与 `labels(fi)` 可分别调用；无标签推理单独读取前者，`audit` 仅供显式监督/核查，提供参考 CSI、原始系数、来源元数据、身份/种子及物理残差。模型 forward 不接收 `audit` 内容或标签。

读取器另提供 `inputs['uncertainty']['coefficient_covariance']`，支持 `matmul()` 与 `modes(count=...)`。该不确定性接口同时给出D1/D2采样DPSS基、系数形状与q优先/延迟次之的展开顺序，供调用方把任意系数模式准确变换为CSI模式。每个worker自行打开读取器，协方差接口只在读取器打开时有效。compact-v2读取器拒绝其他协议、未完成数据和共享模型身份不匹配的数据。当前训练与比较设置见 [ICR-MMSE-Net](icr_mmse.md) 和 [均衡算法对比](equalizer_comparison.md)。

## 运行与断点

已验收数据直接复用，以下命令仅供复现生成流程。

```powershell
python -m pytest -q --basetemp=outputs/runs/pytest_local
python -m data_generation.scripts.fetch_watermark --workers 8
python -m data_generation.scripts.pdd all --smoke --device cpu --threads 4 --run-id ncs1_compact --watermark data/raw/watermark/compact_cpu_smoke.zip
python -m data_generation.scripts.pdd all --smoke --config configs/data/nof1_pddp.json --device cpu --threads 4 --run-id nof1_compact --watermark data/raw/watermark/compact_cpu_smoke.zip
```

`fetch_watermark`仅从FFI官方WatermarkV1链接下载，分段续传、校验HTTP范围/长度和两信道120个ZIP成员CRC，保存源包SHA256。完整源包放在`data/raw/watermark/WatermarkV1.zip`。本地只用少量记录和CPU；smoke保留正式DPSS及7个SNR，但每集合一个窗口，测试两个重复，结果明确标记smoke。

本地验证并提交推送后，服务器`ssh lab448`，目录`/home/fay/JimYang/EPNet`。先检查GPU/内存/磁盘，再执行：

```bash
git pull --ff-only
.venv/bin/python -m data_generation.scripts.fetch_watermark --workers 8
.venv/bin/python -m data_generation.scripts.pdd all --config configs/data/ncs1_pddp.json --device cuda --threads 2 --workers 4 --run-id ncs1_compact_YYYYMMDD
.venv/bin/python -m data_generation.scripts.pdd all --config configs/data/nof1_pddp.json --device cuda --threads 2 --workers 4 --run-id nof1_compact_YYYYMMDD
```

阶段为`audit`、`calibrate`、`generate`、`accept`、`all`。`all`包含验收，`generate`使用已有冻结先验并包含验收。新入口没有均衡评测阶段。相同run ID必须匹配协议、提交、配置和源包；逐窗口及逐样本先flush再写完成标记，公共谱、训练统计和CSI整理阶段也有提交标记。重复运行原命令会继续未完成部分，完成部分保持不变。

当前已验收目录为 `data/processed/nof1_compact_20261026_final/` 与 `data/processed/ncs1_compact_20261026_nmse/`，两者分别包含固定的 train、validation、test 和共享 model 文件。后续算法从这些目录读取估计 CSI，数据来源与历史处理记录保存在各自清单中。

`--workers`仅控制CPU物理窗口预处理（默认1，最多8），使用spawn进程和有界预取；联合CSI求解继续使用所选CPU/GPU后端，所有HDF5文件保持单进程写入。`--threads`限制每进程的数值库线程。应根据内存与CPU实况设置并行数；串行和并行物理链路的全部落盘数组有逐位一致回归测试。

## 验收与交付

验收从落盘系数和完整参考矩阵重算所有D的NMSE，使用精确Fourier/Gram恒等式避免逐个构造大型矩阵，并对每SNR首/末帧额外直接构造完整矩阵核对。每个目标误差≤0.02dB，验证单调和先陡后缓、区间、数量、集合互斥、窗口边界、标签、重复独立性、载波及导频消除。全部训练gamma分子/分母、投影残余和物理残差统计重新计算。测试覆盖小规模显式协方差求解、Hermitian/PSD、模式提取及断点续跑逐数组一致性。

`outputs/runs/<run-id>/`保存`source_manifest.json`、`validation.json`（CSI估计器校验）、`noise_validation.json`、`acceptance.json`、`report.html`、NMSE–SNR图和每个SNR固定测试窗口的参考/估计/误差CSI图。热力图统一`10log10(|H|²/固定参考峰值²)`，色标-80到0dB，D1/D2和各SNR使用相同参考峰值。

完整数据和原始源包留在服务器；验收报告及图片带回本地。完成验收后可删除本轮可再生的`data/interim/<run-id>_calibration.h5`，保留冻结先验、来源与验收报告。数据、缓存和结果均受.gitignore排除，不提交Git。
