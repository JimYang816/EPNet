"""Publication figure of ICR-MMSE-Net: unfolded layers and both sub-modules.

Run from the repository root: python -m scripts.visualize.icr_mmse_paper_figure
Panel (a) is the unfolded joint iteration, (b) the shared element-aligned CSI
refinement, and (c) one MMSE unfolding layer. Every size printed in the figure
is read from configs/training/icr_mmse.json and the classes in
src/algorithms/icr_mmse/model.py. Matrix shading and the constellation are
schematic sketches with a fixed seed; no data or checkpoint is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath

from src.algorithms.icr_mmse.model import FEEDBACK_SLICES, CSICorrector, MMSEUnfoldedLayer

MM = 1 / 25.4
WIDTH, HEIGHT = 182.0, 122.0  # IEEE double-column width, millimetres

INK, MUTED, GRID = "#1F2328", "#5A6270", "#9AA3AE"
CSI_FILL, CSI_EDGE, CSI_DARK = "#E4EEF8", "#3D6FA8", "#2B5788"
MMSE_FILL, MMSE_EDGE, MMSE_DARK = "#E3F1EA", "#2E8059", "#21603F"
SOFT_FILL, SOFT_EDGE = "#FCEBD9", "#C46A1C"
PROD_FILL, PROD_EDGE = "#EEE6F4", "#76579B"
AUX_FILL, AUX_EDGE = "#EFF1F3", "#7B8590"
TRAIN = "#B23A3A"


def model_sizes(config_path: Path) -> dict:
    model = json.loads(config_path.read_text(encoding="utf-8"))["model"]
    bandwidth, window, center = model["bandwidth"], model["window_size"], model["center_size"]
    corrector, equalizer = CSICorrector(bandwidth), MMSEUnfoldedLayer(window, center)
    per_offset = max(item.stop for item in FEEDBACK_SLICES.values())
    count = lambda module: sum(p.numel() for p in module.parameters())
    return dict(B=bandwidth, K=corrector.width, W=window, S=center,
                per_offset=per_offset, row_extra=corrector.conv1.in_channels - per_offset * corrector.width,
                c_in=corrector.conv1.in_channels, c_hidden=corrector.conv1.out_channels,
                c_out=corrector.head.out_channels, mlp_in=equalizer.hidden.in_features,
                mlp_hidden=equalizer.hidden.out_features, mlp_out=equalizer.classifier.out_features,
                corrector_params=count(corrector), noise_mode=model["noise_mode"])


class Canvas:
    """Millimetre drawing helpers; every text artist is kept for layout checks."""

    def __init__(self):
        self.fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1), xlim=(0, WIDTH), ylim=(0, HEIGHT))
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
        self.texts, self.boxes = [], []
    def text(self, x, y, value, size=6.5, color=INK, ha="center", va="center", **kw):
        artist = self.ax.text(x, y, value, fontsize=size, color=color, ha=ha, va=va,
                              zorder=kw.pop("zorder", 6), **kw)
        self.texts.append(artist)
        return artist

    def extent(self, artist):
        """Data-coordinate bounding box of a drawn artist."""
        box = artist.get_window_extent(self.fig.canvas.get_renderer())
        return box.transformed(self.ax.transData.inverted())

    def box(self, x, y, w, h, fill="white", edge=INK, lw=.6, radius=.8, dashed=False, zorder=2):
        self.boxes.append((x, y, x + w, y + h))
        style = f"round,pad=0,rounding_size={radius}" if radius else "square,pad=0"
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=fill,
                                         edgecolor=edge, linewidth=lw, zorder=zorder,
                                         linestyle=(0, (3, 2)) if dashed else "solid"))

    def cell(self, x, y, w, h, fill, edge="white", lw=.25, zorder=3):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                                    linewidth=lw, zorder=zorder))

    def line(self, points, color=INK, lw=.6, dashed=False, dotted=False, zorder=3):
        xs, ys = zip(*points)
        style = (0, (1, 1.2)) if dotted else (0, (3, 2)) if dashed else "solid"
        self.ax.plot(xs, ys, color=color, linewidth=lw, linestyle=style,
                     solid_capstyle="butt", zorder=zorder)

    def arrow(self, points, color=INK, lw=.7, dashed=False, zorder=4):
        """Polyline with a filled head; dashed shafts keep a solid head."""
        points = [tuple(p) for p in points]
        if dashed:
            (x0, y0), (x1, y1) = points[-2], points[-1]
            length = max(np.hypot(x1 - x0, y1 - y0), 1e-9)
            back = min(1.6, .6 * length)
            joint = (x1 - back * (x1 - x0) / length, y1 - back * (y1 - y0) / length)
            self.line(points[:-1] + [joint], color=color, lw=lw, dashed=True, zorder=zorder)
            points = [joint, points[-1]]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
        self.ax.add_patch(FancyArrowPatch(path=MplPath(points, codes),
                                          arrowstyle="-|>,head_length=2.6,head_width=1.15",
                                          color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                                          joinstyle="miter", capstyle="butt", zorder=zorder))

    def operator(self, x, y, kind, r=1.15, color=INK):
        self.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color,
                                 linewidth=.6, zorder=5))
        if kind == "+":
            self.line([(x - .62 * r, y), (x + .62 * r, y)], color=color, lw=.6, zorder=6)
            self.line([(x, y - .62 * r), (x, y + .62 * r)], color=color, lw=.6, zorder=6)
        else:
            self.ax.add_patch(Circle((x, y), .2 * r, facecolor=color, edgecolor="none", zorder=6))

    def badge(self, x, y, label, color):
        self.ax.add_patch(Circle((x, y), 1.45, facecolor=color, edgecolor="none", zorder=5))
        self.text(x, y - .05, label, size=5.9, color="white", weight="bold")


CARDS = [(28.0, "Layer 1"), (74.0, "Layer $t{+}1$"), (120.0, "Layer $T$")]
CARD_W, CARD_Y, CARD_H = 34.0, 92.0, 17.0
FOCUS = 1  # the card enlarged in panels (b) and (c)
PANEL_B, PANEL_C = (3.0, 6.0, 100.0, 76.0), (106.0, 6.0, 73.0, 76.0)  # x, y, w, h


def panel_overview(c: Canvas, s: dict) -> None:
    c.text(3, 119, r"$\bf (a)$  Unfolded joint CSI refinement and equalization",
           size=7.4, ha="left")
    # Legend: colours encode module type, dashes mark supervision used only in training.
    x = 97.0
    for fill, edge, label in ((CSI_FILL, CSI_EDGE, "CSI refinement (shared CNN)"),
                              (MMSE_FILL, MMSE_EDGE, "MMSE unfolding (per layer)")):
        c.box(x, 117.9, 3.2, 2.2, fill=fill, edge=edge, lw=.55, radius=.4)
        x = c.extent(c.text(x + 4.2, 119, label, size=6, ha="left")).x1 + 3
    c.arrow([(x, 119), (x + 5, 119)], color=TRAIN, lw=.6, dashed=True)
    c.text(x + 6.2, 119, "training only", size=6, ha="left")

    # Receiver-side inputs and the deterministic initial state.
    c.box(3, CARD_Y, 21, CARD_H, fill=AUX_FILL, edge=AUX_EDGE, lw=.6)
    c.text(13.5, 106.4, "Receiver inputs", size=6.5, weight="bold")
    c.text(13.5, 102.6, r"$y,\ H_0,\ x_K$", size=7)
    c.text(13.5, 99.3, r"$\sigma^2_{\rm RF},\ C$", size=7)
    c.line([(5, 97.2), (22, 97.2)], color=GRID, lw=.4)
    c.text(13.5, 94.7, r"$\mu_0=x_K,\ v_0=m_D$", size=6.4)

    mid = CARD_Y + 7.0
    for index, (x0, title) in enumerate(CARDS):
        focus = index == FOCUS
        c.box(x0, CARD_Y, CARD_W, CARD_H, fill="#F7F9FB" if focus else "white",
              edge=INK if focus else GRID, lw=1.0 if focus else .55, radius=1.2)
        c.text(x0 + CARD_W / 2, 106.1, title, size=6.6, weight="bold" if focus else "normal")
        c.box(x0 + 2.5, 95.0, 11.5, 8.0, fill=CSI_FILL, edge=CSI_EDGE, lw=.6, radius=.6)
        c.text(x0 + 8.25, 100.5, "CSI", size=6.2, color=CSI_DARK)
        c.text(x0 + 8.25, 97.3, r"$\mathcal{R}_{\phi}$", size=7.2)
        c.box(x0 + 20.0, 95.0, 11.5, 8.0, fill=MMSE_FILL, edge=MMSE_EDGE, lw=.6, radius=.6)
        c.text(x0 + 25.75, 100.5, "MMSE", size=6.2, color=MMSE_DARK)
        layer = ("1", "t{+}1", "T")[index]
        c.text(x0 + 25.75, 97.3, rf"$\mathcal{{M}}_{{\theta_{{{layer}}}}}$", size=7.2)
        c.arrow([(x0 + 14.0, mid), (x0 + 20.0, mid)], lw=.65)
        c.text(x0 + 17.0, mid + 1.6, rf"$H_{{{layer}}}$", size=6)

    # Soft state carried between unfolded layers; dots stand for omitted layers.
    c.arrow([(24.0, mid), (28.0, mid)], lw=.65)
    for x0, x1, label in ((62.0, 74.0, r"$(H,\mu,v)_t$"), (108.0, 120.0, r"$(H,\mu,v)_{t+1}$")):
        c.arrow([(x0, mid), (x1, mid)], lw=.65)
        c.text((x0 + x1) / 2, mid + 2.3, label, size=6)
        c.text((x0 + x1) / 2, mid - 2.0, r"$\cdots$", size=7, color=MUTED)
    c.arrow([(154.0, mid), (158.0, mid)], lw=.65)
    c.box(158, CARD_Y, 21, CARD_H, fill=AUX_FILL, edge=AUX_EDGE, lw=.6)
    c.text(168.5, 106.4, "Outputs", size=6.5, weight="bold")
    c.text(168.5, 102.4, r"bits $\hat b$ from $p_T$", size=6.3)
    c.text(168.5, 98.6, r"CSI $\hat H=H_T$", size=6.3)
    c.text(168.5, 94.9, r"$\mu_T,\ v_T$", size=6.3)
    c.text(179.0, 114.6, r"per-layer loss $\frac{1}{T}\sum_{t=1}^{T}(\mathcal{L}_{{\rm CE},t}+\lambda_H\mathcal{L}_{H,t})$",
           size=6, color=TRAIN, ha="right")

    # Observation bus: the same receiver inputs enter every layer.
    c.line([(13.5, CARD_Y + CARD_H), (13.5, 113.0), (CARDS[-1][0] + 8.25, 113.0)], color=MUTED, lw=.55)
    for x0, _ in CARDS:
        c.arrow([(x0 + 8.25, 113.0), (x0 + 8.25, CARD_Y + CARD_H)], color=MUTED, lw=.55)
        c.ax.add_patch(Circle((x0 + 8.25, 113.0), .45, color=MUTED, zorder=5))
    c.text(66.0, 114.6, r"$y,\ H_0,\ x_K,\ \sigma^2_{\rm RF},\ C$ reused by every layer",
           size=5.8, color=MUTED, ha="left")

    # Zoom guides from the focus card to the two detailed panels.
    x0 = CARDS[FOCUS][0]
    for (left, right), (px, py, pw, ph) in (((x0 + 2.5, x0 + 14.0), PANEL_B),
                                             ((x0 + 20.0, x0 + 31.5), PANEL_C)):
        for top, corner in ((left, px), (right, px + pw)):
            c.line([(top, 95.0), (corner, py + ph)], color=GRID, lw=.45, dotted=True, zorder=1)


def blend(color, weight):
    """Mix a hex colour with white; weight 1 keeps the colour."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(1 - weight * (1 - rgb))


# Display labels for FEEDBACK_SLICES, merged where the figure shows one chip.
FEATURE_CHIPS = (
    (("initial",), r"$H_0[k,j]$", AUX_FILL, AUX_EDGE),
    (("current",), r"$H_t[k,j]$", CSI_FILL, CSI_EDGE),
    (("mean", "variance"), r"$\mu_{t,j},\ v_{t,j}$", SOFT_FILL, SOFT_EDGE),
    (("residual_mean",), r"$r_{t,k}\,\mu_{t,j}^{*}$", PROD_FILL, PROD_EDGE),
    (("channel_variance",), r"$H_t[k,j]\,v_{t,j}$", PROD_FILL, PROD_EDGE),
    (("data", "pilot", "valid"), r"masks $m^{\rm D}_j,m^{\rm P}_j,m_{k,d}$", AUX_FILL, AUX_EDGE),
)


def panel_csi(c: Canvas, s: dict) -> None:
    px, py, pw, ph = PANEL_B
    c.box(px, py, pw, ph, edge=CSI_EDGE, lw=.6, radius=1.5, dashed=True, zorder=1)
    c.text(px + 2, py + ph - 3.2, r"$\bf (b)$  Element-aligned CSI refinement $\mathcal{R}_{\phi}$ with per-layer damping $\eta_t$",
           size=7.2, ha="left")

    # Column 1: banded matrix with the aligned row residual and column soft state.
    n, cell, band, k0, j0 = 12, 1.7, 2, 6, 8
    mx, my = 12.0, 37.0
    rng = np.random.default_rng(7)
    rows, cols = np.mgrid[:n, :n]
    magnitude = np.clip(np.exp(-np.abs(cols - rows) / 2.2) + .07 * rng.standard_normal((n, n)), 0, 1)
    for k in range(n):
        for j in range(n):
            inside = abs(j - k) <= band
            fill = (blend(CSI_DARK, .18 + .72 * magnitude[k, j]) if inside
                    else blend("#6E7781", .08 + .55 * magnitude[k, j]))
            c.cell(mx + j * cell, my + (n - 1 - k) * cell, cell, cell, fill)
    top = my + n * cell
    staircase = []
    for k in range(n):  # right edge of the band, then back along the left edge
        y_hi, y_lo = top - k * cell, top - (k + 1) * cell
        staircase += [(mx + min(n, k + band + 1) * cell, y_hi), (mx + min(n, k + band + 1) * cell, y_lo)]
    for k in reversed(range(n)):
        y_hi, y_lo = top - k * cell, top - (k + 1) * cell
        staircase += [(mx + max(0, k - band) * cell, y_lo), (mx + max(0, k - band) * cell, y_hi)]
    c.line(staircase + staircase[:1], color=CSI_DARK, lw=.75, zorder=4)
    c.box(mx, my, n * cell, n * cell, fill="none", edge=MUTED, lw=.5, radius=0, zorder=4)
    row_y = top - (k0 + 1) * cell
    c.box(mx + (k0 - band) * cell, row_y, (2 * band + 1) * cell, cell, fill="none",
          edge=INK, lw=.9, radius=0, zorder=5)
    c.box(mx + j0 * cell, row_y, cell, cell, fill="none", edge=SOFT_EDGE, lw=1.1, radius=0, zorder=6)

    # Column soft state above the matrix (pilots known, nulls empty), residual on the left.
    sy, rx = top + 1.2, mx - 1.2 - cell
    pilots, nulls = {2, 9}, {0, 11}
    confidence = np.clip(.25 + .7 * rng.random(n), 0, 1)
    for j in range(n):
        fill = ("white" if j in nulls else blend(AUX_EDGE, .7) if j in pilots
                else blend(SOFT_EDGE, .15 + .7 * confidence[j]))
        c.cell(mx + j * cell, sy, cell, cell, fill, edge=GRID if j in nulls else "white")
    for k in range(n):
        c.cell(rx, my + (n - 1 - k) * cell, cell, cell, blend(PROD_EDGE, .2 + .55 * rng.random()))
    c.box(mx + j0 * cell, sy, cell, cell, fill="none", edge=SOFT_EDGE, lw=1.1, radius=0, zorder=6)
    c.box(rx, row_y, cell, cell, fill="none", edge=INK, lw=.9, radius=0, zorder=6)
    centre_x, centre_y = mx + (j0 + .5) * cell, row_y + .5 * cell
    c.line([(centre_x, centre_y + .5 * cell), (centre_x, sy)], color=SOFT_EDGE, lw=.5, dotted=True, zorder=5)
    c.line([(rx + cell, centre_y), (mx + (k0 - band) * cell, centre_y)], color=INK, lw=.5, dotted=True, zorder=5)
    c.text(mx + n * cell / 2, sy + cell + 1.6, r"soft state $\mu_{t,j},\ v_{t,j}$", size=6)
    c.text(rx - .7, centre_y, r"$r_{t,k}$", size=6, ha="right")
    c.text(mx + n * cell / 2, my - 2.0, r"$H_t=H_0+\mathcal{B}(E_t)$", size=6.4)
    c.text(mx + n * cell / 2, my - 5.0, r"row $k$, column $j=k+d$", size=5.9, color=MUTED)
    c.arrow([(mx + n * cell, centre_y), (37.0, centre_y)], lw=.65)

    # Column 2: per-offset features in FEEDBACK_SLICES order, then row-wise inputs.
    widths = [sum(FEEDBACK_SLICES[key].stop - FEEDBACK_SLICES[key].start for key in keys)
              for keys, *_ in FEATURE_CHIPS]
    if sum(widths) != s["per_offset"] or s["row_extra"] != 5:
        raise RuntimeError("Figure feature chips no longer match the corrector input layout")
    c.box(37.0, 19.6, 22.0, 46.4, edge=GRID, lw=.55, radius=.8)
    c.text(48.0, 63.7, r"$F_t[k,d],\ d\in[-B,B]$", size=6.4)
    c.line([(38.5, 62.1), (57.5, 62.1)], color=GRID, lw=.4)
    rows = [(label, fill, edge, width) for (_, label, fill, edge), width in zip(FEATURE_CHIPS, widths)]
    rows += [None, (r"$r_{t,k}$", PROD_FILL, PROD_EDGE, 2), ("noise, position", AUX_FILL, AUX_EDGE, 3)]
    y = 59.8
    for row in rows:
        if row is None:
            c.text(48.0, y + .2, "per row (not per offset)", size=5.8, color=MUTED)
        else:
            label, fill, edge, width = row
            c.box(38.3, y - 1.9, 19.4, 3.8, fill=fill, edge=edge, lw=.45, radius=.4, zorder=3)
            c.text(39.0, y, label, size=5.9, ha="left")
            c.text(57.1, y, str(width), size=5.9, ha="right", color=MUTED)
        y -= 3.95
    c.text(48.0, 24.2, rf"${s['per_offset']}K+{s['row_extra']}={s['c_in']}$ channels", size=6)
    c.text(48.0, 21.6, rf"$K=2B+1={s['K']}$", size=5.8, color=MUTED)

    # Column 3: shared three-layer CNN along the receive-row axis.
    c.arrow([(59.0, 48.5), (63.0, 48.5)], lw=.65)
    bars = ((63.0, 2.4, str(s["c_hidden"])), (67.6, 2.4, str(s["c_hidden"])), (72.2, 3.2, r"$3K$"))
    for index, (x0, w, label) in enumerate(bars):
        if index < 2:
            c.cell(x0, 41.0, w, 15.0, blend(CSI_EDGE, .45 + .2 * index), edge=CSI_DARK, lw=.4)
        else:  # complex increment D_t (2K channels) over gate logits a_t (K channels)
            c.cell(x0, 46.0, w, 10.0, blend(CSI_DARK, .85), edge=CSI_DARK, lw=.4)
            c.cell(x0, 41.0, w, 5.0, blend(CSI_EDGE, .3), edge=CSI_DARK, lw=.4)
        c.text(x0 + w / 2, 39.3, label, size=5.9, color=MUTED)
        if index:
            c.arrow([(bars[index - 1][0] + bars[index - 1][1], 48.5), (x0, 48.5)], lw=.55)
    kernels = (CSICorrector(s["B"]).conv1.kernel_size[0], CSICorrector(s["B"]).conv2.kernel_size[0],
               CSICorrector(s["B"]).head.kernel_size[0])
    field = 1 + sum(k - 1 for k in kernels)
    c.text(62.0, 36.0, rf"Conv1D, kernel {kernels[0]}: ${s['c_in']}\to{s['c_hidden']}\to{s['c_hidden']}$, SiLU",
           size=5.9, ha="left")
    c.text(62.0, 32.9, rf"Conv1D, kernel {kernels[2]}: ${s['c_hidden']}\to3K$ $(D_t{{:}}\,2K,\ a_t{{:}}\,K)$",
           size=5.9, ha="left")
    c.text(62.0, 29.8, f"{field}-row receptive field; {s['corrector_params']:,} CNN parameters",
           size=5.9, ha="left")
    c.text(62.0, 26.4, r"per-layer $\eta_t=\sigma(\beta_t)$;  $s_H=\|H_0\|_F/N$", size=5.9, ha="left")

    # Column 4: gate, damping, and accumulation of the band increment.
    c.arrow([(75.4, 51.0), (77.5, 51.0)], lw=.55)
    c.arrow([(75.4, 43.5), (77.5, 43.5)], lw=.55)
    c.box(77.5, 49.2, 6.0, 3.6, fill=blend(CSI_DARK, .85), edge=CSI_DARK, lw=.5, radius=.4)
    c.text(80.5, 51.0, r"$D_t$", size=6.4, color="white")
    c.box(77.5, 41.7, 11.0, 3.6, fill=CSI_FILL, edge=CSI_EDGE, lw=.5, radius=.4)
    c.text(83.0, 43.5, r"$g_t=\sigma(a_t)$", size=6.1)
    c.operator(86.5, 51.0, "dot")
    c.arrow([(83.5, 51.0), (85.35, 51.0)], lw=.55)
    c.arrow([(86.5, 45.3), (86.5, 49.85)], lw=.55)
    c.operator(86.5, 57.5, "+")
    c.arrow([(86.5, 52.15), (86.5, 56.35)], lw=.55)
    c.text(87.5, 54.3, r"$\eta_t s_H m$", size=5.8, ha="left")
    c.arrow([(82.6, 57.5), (85.35, 57.5)], lw=.55)
    c.text(82.3, 57.5, r"$E_t$", size=6.2, ha="right")
    c.arrow([(86.5, 58.65), (86.5, 61.7)], lw=.6)
    c.text(87.4, 60.0, r"$E_{t+1}$", size=6.0, ha="left")
    c.box(80.5, 61.7, 12.0, 3.6, fill=AUX_FILL, edge=AUX_EDGE, lw=.5, radius=.4)
    c.text(86.5, 63.5, r"$H_0+\mathcal{B}(\cdot)$", size=6.0)
    c.arrow([(92.5, 63.5), (PANEL_C[0] + 4.0, 63.5)], lw=.7)
    c.text(98.0, 65.2, r"$H_{t+1}$", size=6.2)
    c.arrow([(86.5, 65.3), (86.5, 67.4)], color=TRAIN, lw=.55, dashed=True)
    c.text(86.5, 69.0, r"$\mathcal{L}_H$ vs. $H_{\rm ref}$", size=5.8, color=TRAIN)

    for x0, number, label in ((10.0, "1", "Aligned state"), (37.5, "2", "Features"),
                              (62.5, "3", "Shared CNN"), (80.0, "4", "Gated update")):
        c.badge(x0 + 1.45, 72.6, number, CSI_EDGE)
        c.text(x0 + 3.6, 72.6, label, size=6.2, ha="left", weight="bold", color=CSI_DARK)

    # Legend of the schematic shading.
    items = ((blend(CSI_DARK, .6), "corrected band $|j-k|\\leq B$"),
             (blend("#6E7781", .35), "outside band, kept from $H_0$"),
             (blend(SOFT_EDGE, .6), "data column, shade $1-v_{t,j}$"),
             (blend(AUX_EDGE, .7), "pilot column ($v=0$)"),
             ("white", "null column"),
             (blend(PROD_EDGE, .45), "receive-row residual"))
    for index, (fill, label) in enumerate(items):
        y = 28.0 - 3.3 * index
        c.cell(6.0, y - 1.0, 2.0, 2.0, fill, edge=GRID if fill == "white" else "white", lw=.35)
        c.text(9.2, y, label, size=5.9, ha="left")
    c.text(62.0, 22.0, r"$\mathcal{L}_{H}=\|H_{t+1}-H_{\rm ref}\|_F^2\,/\,\|H_{\rm ref}\|_F^2$ (all $N^2$ entries)",
           size=5.8, color=TRAIN, ha="left")


def panel_mmse(c: Canvas, s: dict) -> None:
    px, py, pw, ph = PANEL_C
    c.box(px, py, pw, ph, edge=MMSE_EDGE, lw=.6, radius=1.5, dashed=True, zorder=1)
    c.text(px + 2, py + ph - 3.2, r"$\bf (c)$  MMSE unfolding layer $\mathcal{M}_{\theta_{t+1}}$",
           size=7.2, ha="left")
    c.text(170.0, 78.8, r"$y,\ x_K,\ \mu_t,\ v_t$", size=6.2)
    c.arrow([(170.0, 77.4), (170.0, 75.5)], lw=.55)

    # Fixed physical operators: every action keeps the complete corrected channel.
    c.box(110, 60.6, 67, 14.9, fill=AUX_FILL, edge=AUX_EDGE, lw=.6)
    c.text(143.5, 73.6, r"Global linear step (complete $H_{t+1}$, no matrix inverse)", size=6.3, weight="bold")
    weight = r"W=D_C^{-1}" if s["noise_mode"] == "diagonal" else r"W=C^{-1}"
    c.text(143.5, 70.3, rf"$y_D=y-H_{{t+1}}x_K,\quad {weight},\quad D_C=\mathrm{{diag}}(C)$", size=6.2)
    c.text(143.5, 66.6, r"$b=m_D\odot H_{t+1}^{\rm H}W y_D,\quad a=m_D\odot H_{t+1}^{\rm H}W H_{t+1}(m_D\odot\mu_t)$",
           size=6.1)
    c.text(143.5, 62.6, r"$\delta=\mathrm{diag}(H_{t+1}^{\rm H}D_C^{-1}H_{t+1})$ (column energy, no Gram)",
           size=5.9, color=MUTED)
    c.arrow([(143.5, 60.6), (143.5, 59.0)], lw=.6)

    c.box(110, 48.3, 67, 10.7, fill=MMSE_FILL, edge=MMSE_EDGE, lw=.6)
    c.text(143.5, 57.4, r"Preconditioned Richardson step, learned $\omega\in(0,2)$", size=6.3,
           weight="bold", color=MMSE_DARK)
    c.text(143.5, 52.1, r"$z=m_D\odot\mu_t+\omega\,\dfrac{b-a-\sigma^2_{\rm RF}\,m_D\odot\mu_t}{\delta+\sigma^2_{\rm RF}}$",
           size=6.4)

    per_carrier = s["mlp_in"] // s["W"]
    c.arrow([(126.0, 48.3), (126.0, 44.2)], lw=.6)
    c.text(125.2, 46.3, r"$z_j$", size=6, ha="right")
    c.arrow([(161.0, 48.3), (161.0, 44.2)], lw=.6)
    c.text(159.6, 46.3, rf"$\mu_t,z,b,a,v_t,\ldots$ ({per_carrier}/carrier)", size=5.8, ha="right")

    # Left branch: analytic QPSK denoiser on the linear estimate.
    c.box(110, 21.5, 32, 22.7, edge=MMSE_EDGE, lw=.6)
    c.text(126.0, 42.4, "QPSK distance logits", size=6.3, weight="bold", color=MMSE_DARK)
    cx, cy, half, a = 126.0, 35.4, 5.6, 3.4
    c.line([(cx - half, cy), (cx + half, cy)], color=GRID, lw=.4)
    c.line([(cx, cy - half), (cx, cy + half)], color=GRID, lw=.4)
    zx, zy = cx + 1.9, cy + 1.3
    c.ax.add_patch(Circle((zx, zy), 1.6, facecolor=blend(SOFT_EDGE, .15), edgecolor=SOFT_EDGE,
                          linewidth=.45, linestyle=(0, (2, 1.5)), zorder=3))
    for sx in (-a, a):
        for sy in (-a, a):
            c.line([(zx, zy), (cx + sx, cy + sy)], color=MUTED, lw=.35, dashed=True, zorder=3)
            c.ax.add_patch(Circle((cx + sx, cy + sy), .5, facecolor=MMSE_DARK, edgecolor="none", zorder=5))
    c.line([(zx - .55, zy - .55), (zx + .55, zy + .55)], color=SOFT_EDGE, lw=.8, zorder=6)
    c.line([(zx - .55, zy + .55), (zx + .55, zy - .55)], color=SOFT_EDGE, lw=.8, zorder=6)
    c.text(zx + 1.9, zy - 1.7, r"$z_j$", size=6, ha="left", color=SOFT_EDGE)
    c.text(cx - a - .9, cy + a + .9, r"$s_a$", size=6, ha="right")
    c.text(126.0, 27.4, r"$\ell^{\rm Q}_{j,a}=-|z_j-s_a|^2/\tau_j$", size=6.2)
    c.text(126.0, 23.5, r"$\tau_j=\gamma\,(\sigma^2_{\rm RF}/\delta_j+\kappa\,\bar v_t)$", size=5.8)

    # Right branch: learned calibration from overlapping subcarrier windows.
    c.box(145, 21.5, 32, 22.7, edge=MMSE_EDGE, lw=.6)
    c.text(161.0, 42.4, "Local window calibrator", size=6.3, weight="bold", color=MMSE_DARK)
    x0, w, y0 = 148.3, 2.2, 35.8
    for j in range(12):
        fill = (blend(MMSE_DARK, .65) if 2 <= j <= 5 else blend(MMSE_EDGE, .3) if 6 <= j <= 9 else "white")
        c.cell(x0 + j * w, y0, w, 2.0, fill, edge=GRID, lw=.3)
    c.box(x0 - .25, y0 - .45, 8 * w + .5, 2.9, fill="none", edge=MMSE_DARK, lw=.7, radius=.3, zorder=5)
    c.box(x0 + 4 * w - .1, y0 - .8, 8 * w + .45, 3.6, fill="none", edge=MMSE_EDGE, lw=.55,
          radius=.3, dashed=True, zorder=5)
    c.text(x0 + 4 * w, 40.0, r"$W$", size=6)
    c.text(x0 + 3.5 * w, 33.6, r"$S$", size=6)
    c.text(161.0, 30.7, rf"window $W={s['W']}$, centres $S={s['S']}$", size=5.8)
    c.box(148.0, 25.6, 26.0, 3.4, fill=MMSE_FILL, edge=MMSE_EDGE, lw=.5, radius=.4)
    c.text(161.0, 27.3, rf"tanh MLP ${s['mlp_in']}\to{s['mlp_hidden']}\to{s['mlp_out']}$", size=6.0)
    c.text(161.0, 23.4, r"$\ell^{\rm W}_{j,a}$: $S\times4$ logits per window", size=5.8)

    # Merge of both logits, soft symbols for the next layer, and training loss.
    c.operator(143.5, 18.6, "+")
    c.arrow([(126.0, 21.5), (126.0, 18.6), (142.35, 18.6)], lw=.6)
    c.arrow([(161.0, 21.5), (161.0, 18.6), (144.65, 18.6)], lw=.6)
    c.text(133.0, 20.0, r"$\ell^{\rm Q}$", size=6)
    c.text(154.0, 20.0, r"$\ell^{\rm W}$", size=6)
    c.arrow([(143.5, 17.45), (143.5, 16.6)], lw=.6)
    c.box(110, 7.0, 55, 9.6, fill=SOFT_FILL, edge=SOFT_EDGE, lw=.6)
    c.text(137.5, 14.3, r"$p_{t+1,j,a}=m_{D,j}\,\mathrm{softmax}_a(\ell^{\rm Q}_{j,a}+\ell^{\rm W}_{j,a})$", size=6.3)
    c.text(137.5, 9.8, r"$\mu_{t+1,j}=\sum_a p_{t+1,j,a}\,s_a+x_{K,j},\quad v_{t+1,j}=(1-|\mu_{t+1,j}|^2)\,m_{D,j}$",
           size=6.0)
    c.arrow([(165.0, 12.0), (168.3, 12.0)], color=TRAIN, lw=.55, dashed=True)
    c.text(172.8, 13.4, r"$\mathcal{L}_{\rm CE}$", size=6, color=TRAIN)
    c.text(172.8, 10.7, r"vs. $x_D$", size=5.8, color=TRAIN)


def check_layout(c: Canvas) -> None:
    """Fail on text outside the figure, overlapping text, or text crossing a box border."""
    c.fig.canvas.draw()
    boxes = [(artist, c.extent(artist)) for artist in c.texts]
    for artist, box in boxes:
        if box.x0 < 0 or box.y0 < 0 or box.x1 > WIDTH or box.y1 > HEIGHT:
            raise RuntimeError(f"Text exceeds figure: {artist.get_text()}")
    clashes = [(a.get_text(), b.get_text()) for i, (a, p) in enumerate(boxes) for b, q in boxes[i + 1:]
               if min(p.x1, q.x1) - max(p.x0, q.x0) > .15 and min(p.y1, q.y1) - max(p.y0, q.y0) > .15]
    tol = .1
    for artist, p in boxes:
        for x0, y0, x1, y1 in c.boxes:
            overlaps = min(p.x1, x1) - max(p.x0, x0) > tol and min(p.y1, y1) - max(p.y0, y0) > tol
            inside = p.x0 >= x0 - tol and p.x1 <= x1 + tol and p.y0 >= y0 - tol and p.y1 <= y1 + tol
            if overlaps and not inside:
                clashes.append((artist.get_text(), f"box {x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}"))
    if clashes:
        raise RuntimeError("Overlapping text: " + "; ".join(f"{a!r} / {b!r}" for a, b in clashes))


LATEX = r"""% Generated by scripts/visualize/icr_mmse_paper_figure.py
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/icr_mmse_architecture.pdf}
    \caption{Architecture of ICR-MMSE-Net. (a) $T$ unfolded joint layers. Each layer first refines the
    CSI with $\mathcal{R}_{\phi}$, whose CNN weights $\phi$ are shared by all layers, and then updates the
    soft symbols with a layer-specific MMSE unfolding module $\mathcal{M}_{\theta_t}$, starting from the
    uniform QPSK data prior ($\mu_0=x_K$, $v_0=m_D$, $E_0=0$). (b) For receive row $k$ and offset
    $d\in[-B,B]$, the entry $(k,j=k+d)$ of the banded channel is paired with the residual $r_{t,k}$ and
    the soft state of column $j$, giving $<PER>K+<ROW>=<CIN>$ input channels for $B=<B>$; the features use
    $H/s$ and $y/s$ with $s=\|H_0\|_F/\sqrt{N}$. The gated increment, damped by the per-layer $\eta_t$,
    updates only the band $|j-k|\le B$, and all out-of-band entries of $H_0$ are kept.
    (c) With the corrected channel $H_{t+1}$, a learned preconditioned Richardson step on the LMMSE
    normal equations gives $z$. QPSK distance logits with temperature $\tau_j$
    ($\gamma=\min(\max(e^{q},0.01),100)$, $\kappa=\mathrm{softplus}(u)$, $\bar v_t$ the mean data-carrier
    variance) are added to the logits of a windowed MLP calibrator that reads <PERCARRIER> features per carrier
    ($\mu_t$, $z$, $b$ and $a$ scaled by $1/(\delta+\sigma^2_{\rm RF})$, $v_t$,
    $\log(1+\sigma^2_{\rm RF}/\delta)$, three masks and the carrier position).
    $\theta_t=\{\omega,\gamma,\kappa,\mathrm{MLP}\}$ is specific to each layer. Dashed red paths are used
    only in training: the combined loss is used for joint fine-tuning, while equalizer warm-up and
    refinement pre-training use the CE and NMSE terms alone. Matrix shading is schematic.}
    \label{fig:icr_mmse_architecture}
\end{figure*}
"""


def write_latex(path: Path, s: dict) -> None:
    values = dict(PER=s["per_offset"], ROW=s["row_extra"], CIN=s["c_in"], B=s["B"],
                  PERCARRIER=s["mlp_in"] // s["W"])
    text = LATEX
    for key, value in values.items():
        text = text.replace(f"<{key}>", str(value))
    path.write_text(text, encoding="utf-8")


def draw(output: Path, config: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    sizes = model_sizes(config)
    canvas = Canvas()
    panel_overview(canvas, sizes)
    panel_csi(canvas, sizes)
    panel_mmse(canvas, sizes)
    check_layout(canvas)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "icr_mmse_architecture"
    canvas.fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": "ICR-MMSE-Net architecture"})
    canvas.fig.savefig(stem.with_suffix(".svg"))
    canvas.fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(canvas.fig)
    write_latex(output / "latex_include.tex", sizes)
    for path in [stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png")] + [output / "latex_include.tex"]:
        print(path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/icr_mmse_paper"))
    parser.add_argument("--config", type=Path, default=Path("configs/training/icr_mmse.json"))
    arguments = parser.parse_args()
    draw(arguments.output, arguments.config)
