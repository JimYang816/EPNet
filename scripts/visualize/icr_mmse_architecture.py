"""Draw a concise, manuscript-style overview of ICR-MMSE-Net.

Run from the repository root: python -m scripts.visualize.icr_mmse_architecture
The diagram follows src/algorithms/icr_mmse/model.py. It abstracts the shared
CSI correction, stage-specific MMSE update, and soft-state feedback without
showing implementation details. No data or trained model is required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath


def draw(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    # Double-column width; restrained type and line weights survive reduction.
    fig = plt.figure(figsize=(7.2, 2.30), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 16), ylim=(0, 5.11))
    ax.set_axis_off()
    ink = "#222222"
    texts = []

    def txt(x, y, value, size=10, **kw):
        artist = ax.text(x, y, value, fontsize=size, ha="center", va="center",
                         color=ink, zorder=4, **kw)
        texts.append(artist)

    def box(x, y, w, h, fill="white", edge=ink, dashed=False):
        patch = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                          linewidth=.75,
                          linestyle=(0, (4, 3)) if dashed else "solid",
                          zorder=1)
        ax.add_patch(patch)

    def arrow(points):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", color=ink,
                                    linewidth=.8, mutation_scale=7,
                                    capstyle="butt", joinstyle="miter", zorder=2))

    # A single stage represents the repeated joint refinement, not a second
    # detailed view of the same architecture.
    box(3.20, .62, 10.75, 4.02, edge="#737373", dashed=True)
    txt(8.575, 4.18, r"Joint refinement and equalization ($t=1,\ldots,T$)", 10)

    y, height, cy = 2.28, 1.42, 2.99
    box(.45, y, 2.35, height, fill="#F5F5F5")
    txt(1.625, 3.22, "Initialization", 10)
    txt(1.625, 2.73, r"$\mu_0,\ v_0$", 11)
    txt(1.625, 4.18, r"$y,\ H_0$", 11.4)
    arrow([(1.625, 3.96), (1.625, 3.70)])

    # Each block has only its function and, where helpful, its parameter set.
    box(3.65, y, 2.75, height, fill="#EDF2F7")
    txt(5.025, 3.22, "CSI refinement", 10.2)
    txt(5.025, 2.73, r"$f_{\theta}$", 11.6)
    box(7.20, y, 2.75, height, fill="#EDF4F1")
    txt(8.575, 3.22, "MMSE unfolding", 10.2)
    txt(8.575, 2.73, r"$g_{\phi_t}$", 11.6)
    box(10.75, y, 2.75, height)
    txt(12.125, cy, "Soft-symbol\nestimation", 10.2, linespacing=1.35)

    arrow([(2.80, cy), (3.65, cy)])
    arrow([(6.40, cy), (7.20, cy)])
    txt(6.80, 3.36, r"$H_t$", 10.6)
    arrow([(9.95, cy), (10.75, cy)])
    txt(10.35, 3.36, r"$p_t$", 10.6)
    arrow([(13.50, cy), (14.35, cy)])
    txt(15.02, cy, r"$p_T,\ H_T$", 11.4)

    # Soft means and variances are carried to the following unfolded stage.
    arrow([(12.125, 2.28), (12.125, 1.12), (5.025, 1.12), (5.025, 2.28)])
    txt(8.575, 1.50, r"Soft feedback $(\mu_t, v_t)$", 10)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in texts:
        bounds = artist.get_window_extent(renderer)
        if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
            raise RuntimeError(f"Text exceeds figure bounds: {artist.get_text()}")

    stem = output / "icr_mmse_architecture"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": "ICR-MMSE-Net architecture"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    for suffix in ("pdf", "svg", "png"):
        print(stem.with_suffix(f".{suffix}").resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/icr_mmse"))
    draw(parser.parse_args().output)
