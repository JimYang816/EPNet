"""Independent five-algorithm report, validated baseline/NMSE cache reuse."""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generation.src.transmitter.ofdm import decisions
from src.algorithms.udnet.checkpoint import load_model, validate_dataset, INFERENCE_CODE
from src.algorithms.udnet.inference import equalize_batch
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.data.udnet import WindowPlan, LocalPreprocessor
from src.evaluation import comparison, fingerprints as fp
from src.evaluation.inputs import prepare_batch
from src.evaluation.result_cache import CacheEntry, DEFAULT_CACHE, FileLock

LOG = logging.getLogger(__name__)


def score_udnet(dataset, frames, d, result):
    """Independent bit/SER scoring; labels never enter preprocessing/inference."""
    bits = dataset._h["samples/bits"][frames, d]
    detected = decisions(result.symbols)
    if detected.shape != bits.shape or not np.isfinite(result.symbols).all():
        raise ValueError("Invalid UDNet detection result")
    different = detected != bits
    return different.sum(axis=(1, 2)), different.any(axis=-1).sum(axis=-1)


def _evaluate(dataset_path, checkpoint, run, backend, *, batch_size=8, window_batch_size=2000,
              cache_dir=None, algorithm_parameters=None):
    run, checkpoint = Path(run), Path(checkpoint)
    run.mkdir(parents=True, exist_ok=True)
    model, payload = load_model(checkpoint, backend.device)
    checkpoint_sha = file_sha256(checkpoint)
    # The existing evaluator rehashes all baseline dependencies and checks every
    # cached block. It recomputes only the invalid algorithm/covariance/NMSE part.
    with CompactDataset(dataset_path) as ds:
        validate_dataset(payload, ds)
        if ds.split != "test":
            raise ValueError("UDNet formal evaluation requires the test split")
        if not ds._config.get("smoke") and (ds.frame_count != 600 or len(ds._indices) != 512):
            raise ValueError("Formal UDNet evaluation requires all 600 frames and 512 data carriers")
    baseline = comparison.evaluate(dataset_path, run/"baselines", backend, batch_size=batch_size,
                                   cache_dir=cache_dir, algorithm_parameters=algorithm_parameters)
    with CompactDataset(dataset_path) as ds:
        validate_dataset(payload, ds)
        guard = fp.DataIdentity({name: ds.path.parent/name for name in ("test.h5", "model.h5")})
        # Training and test records must also remain disjoint at evaluation.
        import h5py
        with h5py.File(ds.path.parent/"train.h5", "r") as train:
            if np.intersect1d(ds._h["windows/record"][:], train["windows/record"][:]).size:
                raise ValueError("Training/test record leakage")
        shape = (len(ds.snr_db), ds.frame_count, 2)
        scientific_identity = dict(schema="udnet-test-v1", data=guard.files, checkpoint_sha256=checkpoint_sha,
            code=fp.source_fingerprint(INFERENCE_CODE+fp.INPUT_CODE+("src.evaluation.udnet:score_udnet",
                "data_generation.src.transmitter.ofdm:decisions")), environment=fp.numerical_environment(backend))
        plan = WindowPlan.build(ds._config["nfft"], ds._indices)
        preprocessor = LocalPreprocessor(plan, ds._shared["unit_rf_noise_covariance"][:], backend.device)
        origin = fp.generation_origin(run, backend, batch_size)
        with CacheEntry(Path(cache_dir) if cache_dir else DEFAULT_CACHE, "algorithms/udnet", scientific_identity,
                dict(bit_errors=(shape, "int64"), symbol_errors=(shape, "int64")), shape, origin,
                limits={"bit_errors": (0, 2*len(ds._indices)), "symbol_errors": (0, len(ds._indices))},
                data_guard=guard.assert_unchanged) as entry:
            for si, snr in enumerate(ds.snr_db):
                for d in range(2):
                    for start in range(0, ds.frame_count, batch_size):
                        frames = [fi for fi in range(start, min(start+batch_size, ds.frame_count))
                                  if not entry.complete[si, fi, d]]
                        if not frames:
                            continue
                        batch = prepare_batch(ds, si, frames, d, backend)
                        result = equalize_batch(batch, backend, model=model, preprocessor=preprocessor,
                                                window_batch_size=window_batch_size)
                        bit_errors, symbol_errors = score_udnet(ds, frames, d, result)
                        for j, fi in enumerate(frames):
                            entry.commit((si, fi, d), bit_errors=bit_errors[j], symbol_errors=symbol_errors[j])
                LOG.info("%s UDNet test SNR=%g complete", ds.channel, snr)
                write_json(run/"progress.json", dict(channel=ds.channel, snr_db=float(snr),
                           completed_blocks=int(entry.complete.sum()), total_blocks=int(np.prod(shape))))
            bit_errors = entry.read("bit_errors").sum(axis=(1, 2))
            symbol_errors = entry.read("symbol_errors").sum(axis=(1, 2))
            bits = ds.frame_count*2*len(ds._indices)*2
            symbols = bits//2
            provenance = entry.provenance()
            entry.publish()
        guard.assert_unchanged()
        if file_sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError("Checkpoint changed during evaluation")
        report = dict(baseline)
        report["schema"] = "udnet-independent-five-equalizers-v1"
        report["curves"] = dict(baseline["curves"], UDNet=dict(identifier="udnet", errors=bit_errors.tolist(),
            total_bits=[bits]*len(ds.snr_db), ber=(bit_errors/bits).tolist(),
            symbol_errors=symbol_errors.tolist(), total_symbols=[symbols]*len(ds.snr_db),
            ser=(symbol_errors/symbols).tolist(), provenance=provenance))
        report["udnet"] = dict(checkpoint=str(checkpoint.resolve()), sha256=checkpoint_sha,
            model_config=payload["model_config"], preprocessing=payload["preprocessing"],
            training_identity=payload["identity"], best_epoch=payload["epoch"],
            best_validation_loss=payload["best_loss"], batch_size=payload["batch_size"],
            training_scope=payload["training_scope"], paper_doi=payload["paper_doi"], commit=payload["commit"],
            selection="train/validation only; sum of 22 layer CEs per valid window label",
            validation_metrics=payload["history"][-1].get("validation_metrics"),
            csi_nmse="shared input CSI metric for all five algorithms")
        history = json.loads((checkpoint.parent/"history.json").read_text(encoding="utf-8"))
        plot_report(report, history, run)
        write_json(run/"metrics.json", report)
        write_html(report, run)
        return report


def plot_report(report, history, run):
    snr = np.asarray(report["snr_db"])
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8.1, 5.4), layout="constrained")
        for name, color, marker in zip(report["curves"], ("#667085", "#2477b3", "#df8028", "#15845b", "#b42a55"),
                                       ("o", "s", "^", "D", "*")):
            curve = report["curves"][name]
            values = np.array(curve["ber"])
            ax.semilogy(snr, np.where(values > 0, values, np.nan), marker=marker, label=name,
                        color=color, linewidth=2 if name == "UDNet" else 1.5)
            zero = values == 0
            if zero.any():
                ax.scatter(snr[zero], .5/np.asarray(curve["total_bits"])[zero],
                           marker="v", facecolors="none", edgecolors=color)
        ax.set(xlabel="SNR (dB)", ylabel="Uncoded BER", xticks=snr,
               title=f'{report["channel"]} | {report["frame_count"]} test frames | D1 + D2')
        ax.grid(True, which="both", alpha=.25)
        ax.legend()
        fig.savefig(run/"ber_snr.png", dpi=200)
        plt.close(fig)
        from src.evaluation.comparison_plots import plot_nmse
        plot_nmse(report, run/"nmse_snr.png")
        fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        ser = np.array(report["curves"]["UDNet"]["ser"])
        shown = np.where(ser > 0, ser, .5/np.asarray(report["curves"]["UDNet"]["total_symbols"]))
        ax.semilogy(snr, shown, "*-", color="#b42a55", label="UDNet")
        ax.set(xlabel="SNR (dB)", ylabel="Symbol error rate", xticks=snr, title=report["channel"]+" | QPSK SER")
        ax.grid(True, which="both", alpha=.25)
        ax.legend()
        fig.savefig(run/"ser_snr.png", dpi=200)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train")
        ax.plot([h["epoch"] for h in history], [h["validation_loss"] for h in history], label="Validation")
        ax.axvline(report["udnet"]["best_epoch"], color="#667085", linestyle="--", label="Best validation epoch")
        ax.set(xlabel="Epoch", ylabel="Sum of 22 layer CEs / valid window label", title=report["channel"]+" | Adam training")
        ax.grid(True, alpha=.25)
        ax.legend()
        fig.savefig(run/"training_curves.png", dpi=200)
        plt.close(fig)
        if all("validation_metrics" in h for h in history):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
            for i, value in enumerate(snr):
                rows = [h["validation_metrics"]["per_snr"][i] for h in history]
                axes[0].plot([h["epoch"] for h in history], [r["last_layer_ce"] for r in rows], label=f"{value:g} dB")
                axes[1].semilogy([h["epoch"] for h in history],
                                 [max(r["central_ber"], .5/r["central_bits"]) for r in rows], label=f"{value:g} dB")
            axes[0].set(ylabel="Last-layer cross entropy", title="Validation loss by SNR")
            axes[1].set(ylabel="Central-output BER", title="Validation BER by SNR (zeros at 0.5 / bits)")
            for ax in axes:
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=.25)
                ax.legend(fontsize=8)
            fig.savefig(run/"validation_by_snr.png", dpi=200)
            plt.close(fig)


def write_html(report, run):
    headings = "".join(f"<th>{s:g} dB</th>" for s in report["snr_db"])
    rows = "".join("<tr><th>"+html.escape(name)+"</th>"+"".join(f"<td>{v:.6g}</td>" for v in curve["ber"])+"</tr>"
                   for name, curve in report["curves"].items())
    details = html.escape(json.dumps(report["udnet"], ensure_ascii=False, indent=2))
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{report['channel']} UDNet</title>
<style>body{{font:16px/1.7 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#182c3d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dc;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:16px}}</style>
<h1>{report['channel']}：UDNet 与四基线</h1>
<p>完整测试集每 SNR {report['frame_count']} 帧，D1/D2 共用模型，{report['bits_per_snr']:,} 比特。测试集不参与训练和选模。</p>
<p>复用既有补偿 CSI、划分和 QPSK 映射。32 个连续物理复载波窗、步长 16、两侧各 8 点上下文；仅拼接中心数据槽。局部窗外 ICI 截断。单位噪声协方差白化后，Gram 与 matched 输入共同除以本窗有效数据列的平均能量，ZF 不变；不使用标签或测试集统计量。</p>
<table><tr><th>BER</th>{headings}</tr>{rows}</table>
<img src="ber_snr.png"><img src="nmse_snr.png"><img src="ser_snr.png"><img src="training_curves.png">
<img src="validation_by_snr.png">
<p>NMSE 是五算法共享的输入 CSI 指标：先对全部帧及 D1/D2 的线性 NMSE 求平均，再转 dB。空心倒三角表示零观测错误，显示于 0.5 / 测试总数；SER 的零错误也按此显示。</p>
<p>四基线、Robust EP 训练协方差与 NMSE 均经现有内容和依赖指纹校验后复用或重算。来源和各部分缓存键见 metrics.json。</p>
<pre>{details}</pre></html>'''
    (run/"report.html").write_text(body, encoding="utf-8")


def evaluate(dataset_path, checkpoint, run, backend, **kwargs):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".udnet_evaluation.lock"):
        return _evaluate(dataset_path, checkpoint, run, backend, **kwargs)
