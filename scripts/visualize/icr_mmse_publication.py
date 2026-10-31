"""Draw a concise, vector manuscript figure of the current ICR-MMSE-Net.

Run from the repository root::

    python -m scripts.visualize.icr_mmse_publication

No data or trained model is loaded. The figure describes the default iterative
architecture in src/algorithms/icr_mmse/model.py. It deliberately omits tensor
sizes, training losses, and the algebra internal to the linear update.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath


ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 180.0, 100.0  # millimetres, for a double-column manuscript
INK = "#25303A"
MUTED = "#596775"
HAIR = "#CBD2D9"
NEUTRAL = "#F4F6F8"
AMBER = "#AD7431"
AMBER_DARK = "#795021"
AMBER_FILL = "#FBF3E6"
BLUE = "#377796"
BLUE_DARK = "#23566E"
BLUE_FILL = "#EDF5F8"


class Canvas:
    """Vector geometry in millimetres, with bounds recorded for visual QA."""

    def __init__(self):
        self.fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1), xlim=(0, WIDTH), ylim=(0, HEIGHT))
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
        self.texts = []
        self.boxes = []

    def text(self, x, y, value, size=8.0, color=INK, ha="center", **kwargs):
        item = self.ax.text(x, y, value, fontsize=size, color=color, ha=ha,
                            va="center", linespacing=1.25, zorder=6, **kwargs)
        self.texts.append(item)
        return item

    def box(self, x, y, w, h, fill="white", edge=HAIR, lw=0.75, radius=0.9,
            dashed=False, check=True):
        item = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge, linewidth=lw,
                              linestyle=(0, (3, 2)) if dashed else "solid", zorder=2)
        self.ax.add_patch(item)
        if check:
            self.boxes.append((x, y, x + w, y + h))

    def line(self, points, color=INK, lw=0.8, dashed=False, zorder=3):
        x, y = zip(*points)
        self.ax.plot(x, y, color=color, linewidth=lw,
                     linestyle=(0, (3, 2)) if dashed else "solid",
                     solid_capstyle="butt", zorder=zorder)

    def arrow(self, points, color=INK, lw=0.85, head=4.2, zorder=4):
        path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
        self.ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle=f"-|>,head_length={head},head_width={head * .56}",
            mutation_scale=1, linewidth=lw, color=color, shrinkA=0, shrinkB=0,
            joinstyle="round", capstyle="butt", zorder=zorder))

    def dot(self, x, y, color=INK, radius=0.4):
        self.ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="none", zorder=5))

    def dots(self, x, y):
        for dx in (-1.35, 0, 1.35):
            self.dot(x + dx, y, MUTED, 0.22)

    def check_layout(self):
        """Detect clipped labels, overlapping labels, and labels across box edges."""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inverse = self.ax.transData.inverted()
        boxes = [(text, text.get_window_extent(renderer).transformed(inverse)) for text in self.texts]
        problems = []
        tolerance = 0.10
        for text, rect in boxes:
            if min(rect.x0, rect.y0) < 0.7 or rect.x1 > WIDTH - .7 or rect.y1 > HEIGHT - .7:
                problems.append(f"label outside canvas: {text.get_text()!r}")
            for x0, y0, x1, y1 in self.boxes:
                overlaps = min(rect.x1, x1) - max(rect.x0, x0) > tolerance and min(rect.y1, y1) - max(rect.y0, y0) > tolerance
                inside = rect.x0 >= x0 - tolerance and rect.x1 <= x1 + tolerance and rect.y0 >= y0 - tolerance and rect.y1 <= y1 + tolerance
                if overlaps and not inside:
                    problems.append(f"label crosses box: {text.get_text()!r}")
        for index, (text, rect) in enumerate(boxes):
            for other, other_rect in boxes[index + 1:]:
                if (min(rect.x1, other_rect.x1) - max(rect.x0, other_rect.x0) > tolerance
                        and min(rect.y1, other_rect.y1) - max(rect.y0, other_rect.y0) > tolerance):
                    problems.append(f"overlapping labels: {text.get_text()!r}, {other.get_text()!r}")
        if problems:
            raise RuntimeError("\n".join(problems))


def layer_card(c: Canvas, x, label, highlight=False):
    c.box(x, 73, 28, 16, edge=INK if highlight else HAIR, lw=1.05 if highlight else .75)
    c.text(x + 14, 85.4, label, size=8)
    c.box(x + 2, 75, 10.1, 6.4, fill=AMBER_FILL, edge=AMBER, lw=.7, radius=.55)
    c.box(x + 16, 75, 10.1, 6.4, fill=BLUE_FILL, edge=BLUE, lw=.7, radius=.55)
    c.text(x + 7.05, 78.2, "CSI", size=7.5, color=AMBER_DARK)
    c.text(x + 21.05, 78.2, "MMSE", size=7.3, color=BLUE_DARK)
    c.arrow([(x + 12.1, 78.2), (x + 16, 78.2)], head=3.1, lw=.7)


def overview(c: Canvas):
    c.text(3, 96.5, "(a)  ICR-MMSE-Net", size=9, weight="bold", ha="left")
    c.text(177, 96.5, r"$T$ unfolded joint layers", size=8, ha="right", color=MUTED)

    c.box(3, 73, 28, 16, fill=NEUTRAL)
    c.text(17, 85.4, "Receiver inputs", size=7.6, weight="bold")
    c.text(17, 80.4, r"$y,\ H_0,\ x_K$", size=8.7)
    c.text(17, 76.1, r"$\sigma^2_{\mathrm{RF}},\ C$", size=8.5)
    layer_card(c, 40, "Layer 1")
    layer_card(c, 84, r"Layer $t$", highlight=True)
    layer_card(c, 128, r"Layer $T$")
    c.arrow([(31, 78.2), (40, 78.2)])
    for left, right in ((68, 84), (112, 128)):
        center = (left + right) / 2
        c.line([(left, 78.2), (center - 3.2, 78.2)])
        c.dots(center, 78.2)
        c.arrow([(center + 3.2, 78.2), (right, 78.2)], head=3.8)
    c.arrow([(156, 78.2), (163, 78.2)])
    c.text(170, 84.7, "Outputs", size=7.6, weight="bold")
    c.text(170, 79.1, r"$\widehat b$", size=11)
    c.text(170, 74.4, r"$H_T$", size=9)

    # Receiver inputs are reused by every layer, not re-estimated inside it.
    c.line([(17, 89), (17, 92), (142, 92)], color=MUTED, lw=.65)
    for center in (54, 98, 142):
        c.dot(center, 92, color=MUTED, radius=.3)
        c.arrow([(center, 92), (center, 89)], color=MUTED, head=3.1, lw=.65)

    c.text(3, 67.3, r"Joint state: $s_t=(H_t,\,\mu_t,\,v_t)$", size=7.5, ha="left", color=MUTED)
    c.text(177, 67.3, "Soft information passes between layers", size=7.3, ha="right", color=MUTED)
    c.line([(84, 73), (11, 54)], color=HAIR, lw=.65, dashed=True, zorder=1)
    c.line([(112, 73), (168, 54)], color=HAIR, lw=.65, dashed=True, zorder=1)


def detail(c: Canvas):
    c.text(98, 59.5, r"(b)  Joint layer $t$", size=9, weight="bold")

    c.box(11, 18.5, 74, 35.5, fill=AMBER_FILL, edge=AMBER, lw=.65, radius=1.2)
    c.box(96, 18.5, 72, 35.5, fill=BLUE_FILL, edge=BLUE, lw=.65, radius=1.2)
    c.text(48, 49.9, "CSI refinement", size=8.6, color=AMBER_DARK, weight="bold")
    c.text(132, 49.9, "MMSE unfolding", size=8.6, color=BLUE_DARK, weight="bold")

    # Only four coarse operations are shown; numerical details stay in the caption.
    for x, w, color in ((14, 30, HAIR), (51, 31, AMBER), (99, 29, BLUE), (135, 30, BLUE)):
        c.box(x, 25, w, 20, edge=color, lw=.75, radius=.7)
    c.text(29, 40.6, "Feedback\nconstruction", size=8, weight="bold")
    c.text(29, 31.5, "Residual +\nsoft mean / variance", size=7.3, color=MUTED)
    c.text(66.5, 40.6, "CSI\ncorrection", size=8, weight="bold")
    c.text(66.5, 32.4, r"Shared CNN $\phi$", size=7.7, color=AMBER_DARK)
    c.text(66.5, 28.3, "Gated band update", size=7.1, color=MUTED)
    c.text(113.5, 40.6, "MMSE\nupdate", size=8, weight="bold")
    c.text(113.5, 31.5, "Pilot cancellation +\nlearned linear step", size=7.1, color=MUTED)
    c.text(150, 40.6, "Soft\ndetection", size=8, weight="bold")
    c.text(150, 31.5, "QPSK probabilities +\nlocal calibration", size=7.1, color=MUTED)

    flow_y = 35
    c.arrow([(2, flow_y), (14, flow_y)])
    c.text(6.2, 40.5, r"$s_{t-1}$", size=9)
    c.dot(7.5, flow_y)
    c.arrow([(44, flow_y), (51, flow_y)], head=3.7)
    c.text(47.5, 38.4, r"$F_t$", size=8)
    c.arrow([(82, flow_y), (99, flow_y)])
    c.text(90.5, 38.9, r"$H_t$", size=9.2)
    c.arrow([(128, flow_y), (135, flow_y)], head=3.7)
    c.text(131.5, 38.4, r"$z_t$", size=8)
    c.arrow([(165, flow_y), (178, flow_y)])
    c.text(172.1, 40.5, r"$\mu_t,v_t$", size=8.6)

    c.text(48, 21.6, r"$H_t=H_0+\mathcal{B}(E_t)$", size=9.0, color=AMBER_DARK)
    c.text(132, 21.6, r"Layer-specific parameters $\theta_t$", size=7.2, color=BLUE_DARK)

    # The linear step also consumes the preceding soft state, independently of
    # the refiner. Its route never crosses the main corrected-channel path.
    c.arrow([(7.5, flow_y), (7.5, 11.3), (113.5, 11.3), (113.5, 25)], color=MUTED, lw=.75)
    c.text(60.5, 14.5, r"Previous soft state $(\mu_{t-1},v_{t-1})$", size=7.5, color=MUTED)
    c.text(3, 4.2, r"CNN weights $\phi$ are shared; damping $\eta_t$ and MMSE parameters $\theta_t$ vary across layers.",
           size=7.2, color=MUTED, ha="left")


CAPTION = r"""% Include with graphicx; this is a manuscript snippet, not a standalone document.
% Copy the PDF into your manuscript's figures/ directory.
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/icr_mmse_publication.pdf}
    \caption{Architecture of ICR-MMSE-Net.
    (a) $T$ unfolded layers propagate $s_t=(H_t,\mu_t,v_t)$, starting from compensated
    CSI $H_0$ and a uniform data-symbol prior. Receiver inputs and carrier masks are
    reused across layers. (b) Residual and soft-state features drive a shared CNN
    for gated, band-limited CSI refinement. The full initial channel is retained in
    $H_t=H_0+\mathcal{B}(E_t)$. The corrected CSI drives a learned preconditioned
    Richardson update, followed by QPSK soft detection with local logit calibration.
    Output probabilities yield the new soft mean and variance. CNN weights $\phi$
    are shared; damping $\eta_t$ and equalizer parameters $\theta_t$ are layer-specific.
    Final bit-marginal decisions yield $\widehat b$. Training supervision is omitted.}
    \label{fig:icr_mmse_publication}
\end{figure*}
"""


NOTES = """ICR-MMSE-Net 论文结构图
生成日期：2026-10-28

文件用途
icr_mmse_publication.pdf：180 mm 宽的矢量论文插图，字体嵌入。
icr_mmse_publication.svg：可编辑矢量版本，保留文字。
icr_mmse_publication.png：600 dpi 位图版本。
icr_mmse_publication_preview.png：用于快速预览。
latex_include.tex：英文图注及双栏 LaTeX 插图片段。

绘图取舍
采用“全网展开 + 单层结构”布局，单层只展示四个主要处理块。
颜色区分 CSI 修正与 MMSE 展开，文字和外框在灰度打印时仍能独立识别。
不在图内展开卷积通道数、逐元素特征、窗口切片、损失函数和训练阶段。
图为独立绘制，仅参考下列论文的信息组织方式，未复制其图像。

图中符号与实现
图使用 t=1,...,T 的层号：每层从 s_(t-1) 产生 s_t。
F_t 表示本层由旧状态构造的反馈特征，对应设计文档零起点记号中的 F_(t-1)。
H_0 使用 compensated CSI；H_t=H_0+B(E_t) 保留完整基础矩阵。
CSI CNN 权重跨层共享；门控输出随输入变化；阻尼 eta_t 逐层独立。
MMSE 更新包含导频消除和预条件 Richardson 子步；不是逐层精确求逆。
Soft detection 合并 QPSK 距离 logits、局部校准及 softmax，并计算软均值/方差。
顶层箭头传递 H_t、mu_t、v_t。单层右端标注软状态，H_t 在两模块之间标注。
噪声、导频、掩码等公共侧信息已在总体输入及图注说明，不逐个重复连线。
真实信道和真实符号不进入推理图。

主要参考及具体借鉴
[1] H. He, C.-K. Wen, S. Jin, and G. Y. Li,
    Model-Driven Deep Learning for MIMO Detection,
    IEEE Transactions on Signal Processing, vol. 68, pp. 1702-1715, 2020.
    DOI: https://doi.org/10.1109/TSP.2020.2976585
    作者全文: https://arxiv.org/pdf/1907.09439
    参考 Fig. 2：多层网络上方概览与一个代表层的结构展开。

[2] N. Shlezinger, R. Fu, and Y. C. Eldar,
    DeepSIC: Deep Soft Interference Cancellation for Multiuser MIMO Detection,
    IEEE Transactions on Wireless Communications, vol. 20, no. 2, pp. 1349-1362, 2021.
    DOI: https://doi.org/10.1109/TWC.2020.3032663
    作者全文: https://arxiv.org/pdf/2002.03214
    参考 Fig. 3-4：重复模块、概率信息传递和最终判决边界。

[3] H. Zhao, C. Yang, Y. Xu, F. Ji, M. Wen, and Y. Chen,
    Model-Driven Based Deep Unfolding Equalizer for Underwater Acoustic OFDM Communications,
    IEEE Transactions on Vehicular Technology.
    DOI: https://doi.org/10.1109/TVT.2022.3230143
    作者全文: https://arxiv.org/pdf/2207.04478
    参考 Fig. 4-5：水声 OFDM 的单层 MMSE 展开与分类输出。

[4] K. Kang et al., Mixed-Timescale Deep-Unfolding for Joint Channel Estimation
    and Hybrid Beamforming, IEEE Journal on Selected Areas in Communications,
    vol. 40, no. 9, pp. 2510-2528, 2022.
    DOI: https://doi.org/10.1109/JSAC.2022.3191124
    作者全文: https://arxiv.org/abs/2206.03755
    参考 Fig. 3：联合任务分组和代表层展开，不引入波束成形或训练阶段。

分区依据
IEEE 2025 年 8 月公开 Title List 中，上述 TSP、TWC、TVT、JSAC 均列为 JIF Q1。
https://open.ieee.org/wp-content/uploads/IEEE-Title-List-August-2025-CAPES.pdf
此处“一区”采用可核验的 JCR/JIF Q1 口径，不将其等同于中科院分区。

复现
python -m scripts.visualize.icr_mmse_publication
源码：scripts/visualize/icr_mmse_publication.py
算法依据：src/algorithms/icr_mmse/model.py、docs/icr_mmse_algorithm_design.md。
"""


def draw(output: Path, config: Path):
    settings = json.loads(config.read_text(encoding="utf-8"))
    model = settings["model"]
    if settings.get("input_view") != "compensated":
        raise ValueError("This figure is for the accepted compensated-CSI design")
    if model.get("correction_mode") != "iterative" or not model.get("use_gate") or not model.get("use_uncertainty"):
        raise ValueError("This figure describes the full iterative, gated, uncertainty-aware architecture")
    if model.get("equalizer_repeats", 1) != 1 or model.get("correction_passes", 1) != 1:
        raise ValueError("This figure assumes one correction and one equalizer step per joint layer")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "icr-mmse-publication",
    })
    canvas = Canvas()
    overview(canvas)
    detail(canvas)
    canvas.check_layout()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "icr_mmse_publication"
    canvas.fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": "ICR-MMSE-Net architecture"})
    canvas.fig.savefig(stem.with_suffix(".svg"))
    canvas.fig.savefig(stem.with_suffix(".png"), dpi=600)
    canvas.fig.savefig(output / "icr_mmse_publication_preview.png", dpi=240)
    plt.close(canvas.fig)
    (output / "latex_include.tex").write_text(CAPTION, encoding="utf-8")
    notes = NOTES + (f"\n当前配置：T={model['layers']}，修正半带宽 B={model['bandwidth']}，"
                     f"局部窗口={model['window_size']}，中心输出={model['center_size']}，"
                     f"noise_mode={model['noise_mode']}。\n")
    (output / "figure_notes.txt").write_text(notes, encoding="utf-8")
    print(f"Layout checks passed. Figure: {WIDTH:g} x {HEIGHT:g} mm.")
    for suffix in (".pdf", ".svg", ".png"):
        print(stem.with_suffix(suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures/icr_mmse_publication")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/icr_mmse.json")
    options = parser.parse_args()
    draw(options.output, options.config)
