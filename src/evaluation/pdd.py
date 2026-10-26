"""MMSE and EP evaluation using estimated CSI; reference CSI is for error metrics."""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.experiment import write_json
from data_generation.src.pdd import DATA_SCHEMA, ESTIMATION_MODE, FRAME_STRUCTURE
from data_generation.src.transmitter.ofdm import decisions
from src.algorithms.MMSE import mmse
from src.algorithms.ep import EqualizerSystem, ep
from src.algorithms.ep.system import NoiseWhitening
from src.data.ncs1 import NCS1Dataset

LOG = logging.getLogger(__name__)
CURVES = ("MMSE", "EP")
EQUALIZATION_SCHEMA = "pddp-mmse-ep-equalization-v1"


def ep_inputs(channel, raw_observation, public_training, data_mask):
    """All receive rows, full data-column ICI, known pilot cancellation."""
    return channel[:, data_mask], raw_observation-channel @ public_training


def summarize(errors, nmse, data_count):
    errors, nmse = np.asarray(errors), np.asarray(nmse)
    if errors.ndim != 4 or errors.shape[0] != len(CURVES) or errors.shape[-1] != 2 or nmse.shape != errors.shape[1:]:
        raise ValueError("Expected algorithm x SNR x frame x D errors and SNR x frame x D NMSE")
    count = errors.shape[2]
    denominator = count*2*data_count*2
    curves = {name: dict(errors=errors[j].sum(axis=(1, 2)).tolist(), total_bits=[denominator]*errors.shape[1],
                        ber=(errors[j].sum(axis=(1, 2))/denominator).tolist(),
                        errors_d1_d2=errors[j].sum(axis=1).tolist(), total_bits_per_d=count*data_count*2)
              for j, name in enumerate(CURVES)}
    linear = nmse.mean(axis=(1, 2))
    return dict(curves=curves, frame_count=count, data_symbol_count=count*2,
                mean_nmse_linear=linear.tolist(), mean_nmse_db=(10*np.log10(np.maximum(linear, 1e-300))).tolist(),
                nmse_d1_d2_linear=nmse.mean(axis=1).tolist(),
                nmse_d1_d2_db=(10*np.log10(np.maximum(nmse.mean(axis=1), 1e-300))).tolist())


def evaluate(path, run, backend):
    run = Path(run)
    with NCS1Dataset(path, expected_schema=DATA_SCHEMA) as dataset:
        h, c = dataset.h, dataset.config
        ep_parameters = dict(max_iterations=c["ep_max_iterations"], damping=c["ep_damping"],
                             relative_tolerance=c["ep_relative_tolerance"], variance_floor=c["ep_variance_floor"],
                             min_iterations=c["ep_min_iterations"], convergence_patience=c["ep_convergence_patience"])
        ep_parameters_json = json.dumps(ep_parameters, sort_keys=True, allow_nan=False)
        nsnr, count = len(c["snr_db"]), len(dataset)
        unit_cov, known, mask = h["unit_rf_noise_covariance"][:], h["known_training"][:], dataset.masks["data"]
        prepared_noise = NoiseWhitening(unit_cov, backend)
        checkpoint = run/"equalization.h5"
        resuming = checkpoint.exists()
        out = h5py.File(checkpoint, "r+" if resuming else "w")
        try:
            if resuming:
                if (out.attrs.get("schema") != EQUALIZATION_SCHEMA or out.attrs.get("estimation_mode") != ESTIMATION_MODE
                        or out.attrs.get("dataset_provenance") != h.attrs["provenance_json"]
                        or out.attrs.get("config_hash") != h.attrs["config_hash"]
                        or out.attrs.get("curves_json") != json.dumps(CURVES)
                        or out.attrs.get("ep_parameters_json") != ep_parameters_json):
                    raise ValueError("PDDP evaluation checkpoint provenance mismatch")
            else:
                out.attrs.update(schema=EQUALIZATION_SCHEMA, estimation_mode=ESTIMATION_MODE,
                                 dataset_provenance=h.attrs["provenance_json"], config_hash=h.attrs["config_hash"],
                                 csi_source="estimated", curves_json=json.dumps(CURVES),
                                 ep_parameters_json=ep_parameters_json, complete=False)
                out.create_dataset("bit_errors", (len(CURVES), nsnr, count, 2), dtype="int64")
                out.create_dataset("seconds", (len(CURVES), nsnr, count, 2), dtype=float)
                out.create_dataset("complete", (nsnr, count, 2), dtype=bool)
                out.create_dataset("ep_diagnostics", (nsnr, count, 2), dtype=h5py.string_dtype("utf-8"))
            for i in range(count):
                LOG.info("PDDP equalization frame %d/%d", i+1, count)
                for d, s in enumerate((1, 2)):
                    for k in range(nsnr):
                        if out["complete"][k, i, d]:
                            continue
                        truth = h["data_bits"][i, d]
                        y = h["observations"][k, i, s].astype(complex)
                        estimated = h["estimated_csi"][k, i, d].astype(complex)
                        variance = float(h["rf_noise_variance"][k, i])
                        begin = time.perf_counter()
                        mu = mmse(estimated, y, mask, variance*np.diag(unit_cov).real)
                        out["bit_errors"][0, k, i, d] = np.count_nonzero(decisions(mu) != truth)
                        out["seconds"][0, k, i, d] = time.perf_counter()-begin
                        data_channel, residual = ep_inputs(estimated, y, known[s], mask)
                        begin = time.perf_counter()
                        system = EqualizerSystem(data_channel, unit_cov, backend, prepared_noise=prepared_noise)
                        mu, info = ep(system, residual, variance, **ep_parameters)
                        backend.synchronize()
                        out["bit_errors"][1, k, i, d] = np.count_nonzero(decisions(mu) != truth)
                        out["seconds"][1, k, i, d] = time.perf_counter()-begin
                        out["ep_diagnostics"][k, i, d] = json.dumps(info)
                        out.flush()
                        out["complete"][k, i, d] = True
                        out.flush()
                write_json(run/"evaluation_progress.json", dict(completed_d=int(out["complete"][:].sum()), total_d=nsnr*count*2))
            out.attrs["complete"] = True
            report = summarize(out["bit_errors"][:], h["nmse_linear"][:], c["data_count"])
            ep_stats = [dict(snr_db=c["snr_db"][k], frame=i, d=d+1,
                             **json.loads(out["ep_diagnostics"][k, i, d]))
                        for k in range(nsnr) for i in range(count) for d in range(2)]
            report["timing_seconds_by_curve"] = dict(zip(CURVES, out["seconds"][:].sum(axis=(1, 2, 3)).tolist()))
            report["ep_parameters"] = ep_parameters
            report["ep_summary"] = dict(total=len(ep_stats), unconverged=sum(not x["converged"] for x in ep_stats),
                                        rejected_updates=sum(x["rejected_updates"] for x in ep_stats),
                                        mean_iterations=float(np.mean([x["iterations"] for x in ep_stats])),
                                        stop_reasons={reason: sum(x["stop_reason"] == reason for x in ep_stats)
                                                      for reason in ("relative_tolerance", "max_iterations")})
        except BaseException as exc:
            write_json(run/"evaluation_failure.json", dict(type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            out.close()
        report.update(complete=True, schema=DATA_SCHEMA, equalization_schema=EQUALIZATION_SCHEMA,
                      frame_structure=FRAME_STRUCTURE, channel=c.get("channel", "NCS1"), csi_source="estimated",
                      estimation_mode=ESTIMATION_MODE, feedback_iterations=0, snr_db=c["snr_db"],
                      prior=json.loads(h.attrs["prior_json"]), provenance=json.loads(h.attrs["provenance_json"]),
                      actual_snr_db=h["actual_snr_db"][:].tolist(),
                      noiseless_nmse_d1_d2_linear=h["noiseless_nmse_linear"][:].mean(axis=0).tolist(),
                      projection_nmse_d1_d2_linear=h["projection_nmse_linear"][:].mean(axis=0).tolist(),
                      generation_seconds=float(h["generation_seconds"][:].sum()),
                      physical_diagnostics=[json.loads(x) for x in h["diagnostics_json"][:]])
        report["csi_covariance_floor"] = dict(ratio=c["csi_model_error_ratio"],
            power_source="noise-subtracted public P1/P2 RX power, normalized to D transmit energy",
            thermal_noise_unchanged=True,
            mean_effective_rf_variance_by_snr=h["csi_effective_rf_variance"][:].mean(axis=1).tolist())
        diagnostics = report["physical_diagnostics"]
        report["model_mismatch"] = dict(policy=c.get("model_mismatch_policy", "reject"),
            link_limit_db=c["link_error_db"], support_limit_db=c["support_error_db"],
            frames_with_isi=sum(x.get("model_mismatch_retained", False) for x in diagnostics),
            frames_with_support_omission=sum(not x.get("support_passed", True) for x in diagnostics),
            maximum_isi_ratio_db=float(10*np.log10(max([v for x in diagnostics for v in x.get("intersymbol_leakage", [0])]+[1e-300]))),
            maximum_support_residual_db=float(10*np.log10(max([v for x in diagnostics for v in x.get("support_residuals_p1_d1_d2_p2", [0])]+[1e-300]))))
        write_json(run/"model_mismatch_summary.json", report["model_mismatch"])
        if not c.get("smoke") and (report["frame_count"] != 25 or report["curves"]["MMSE"]["total_bits"] != [51200]*nsnr):
            raise RuntimeError("Formal PDDP evaluation requires 25 frames, 50 D, 51200 bits per SNR per curve")
        write_json(run/"ep_diagnostics.json", ep_stats)
        write_json(run/"metrics.json", report)
        plot_results(h, report, run)
        write_report(report, run)
        return report


def plot_results(h, report, run):
    run = Path(run)
    channel_name = report.get("channel", h.attrs.get("channel", "NCS1"))
    snr = np.asarray(report["snr_db"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color, marker in zip(CURVES, ("#2368a2", "#cd5b31"), ("o", "s")):
        values = report["curves"][name]
        ber = np.asarray(values["ber"])
        ax.semilogy(snr, np.where(ber > 0, ber, np.nan), color=color, marker=marker, label=name)
        zero = ber == 0
        if np.any(zero):
            ax.scatter(snr[zero], .5/np.asarray(values["total_bits"])[zero], marker="v", facecolors="none", edgecolors=color)
    ax.set(xlabel="Mean D1/D2 useful-window SNR (dB)", ylabel="Uncoded BER",
           title=f"{channel_name}: MMSE / EP | estimated CSI")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.text(.5, .015, "Open triangles: zero measured errors; display at 0.5 / tested bits.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(run/"ber_snr.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(snr, report["mean_nmse_db"], "o-", label="Mean over D1/D2")
    ax.set(xlabel="SNR (dB)", ylabel="10 log10(mean linear full-matrix NMSE)",
           title=f"{channel_name}: P-D-D-P joint CSI")
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run/"nmse_snr.png", dpi=180)
    plt.close(fig)
    plot_csi_heatmaps(h, report, run)


def plot_csi_heatmaps(h, report, run):
    """Render CSI with a shared reference-normalized Viridis dB scale."""
    run = Path(run)
    channel_name = report.get("channel", h.attrs.get("channel", "NCS1"))
    snr = np.asarray(report["snr_db"])
    matches = np.flatnonzero((h["record"][:] == 1) & (h["start_seconds"][:] == 2))
    if len(matches) != 1:
        raise ValueError(f"Required {channel_name}_001 @ 2s heatmap frame is missing")
    index = int(matches[0])
    reference = h["reference_csi"][index].astype(complex)
    estimates = h["estimated_csi"][:, index].astype(complex)
    peak = float(np.max(abs(reference)))
    if peak <= 0:
        raise RuntimeError("Zero reference peak for CSI heatmaps")
    maximum = max(peak, float(np.max(abs(estimates))), float(np.max(abs(estimates-reference[None]))))
    vmax = max(0., float(20*np.log10(maximum/peak)))
    n = reference.shape[-1]
    for k, value in enumerate(snr):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for d in range(2):
            for ax, channel, title in zip(axes[d], (reference[d], estimates[k, d], estimates[k, d]-reference[d]),
                                           ("Reference CSI", "Estimated CSI", "Complex difference magnitude")):
                db = 20*np.log10(np.maximum(abs(channel)/peak, 1e-4))
                image = ax.imshow(db, origin="lower", vmin=-80, vmax=vmax, cmap="viridis",
                                  extent=(-n//2, n//2-1, -n//2, n//2-1), interpolation="nearest")
                ax.set(title=f"D{d+1}: {title}", xlabel="Transmitted carrier", ylabel="Received carrier")
        fig.colorbar(image, ax=axes, shrink=.85, label="dB relative to shared reference peak; fixed scale across D and SNR")
        fig.suptitle(f"{channel_name}_001 @ 2s | SNR {value:g} dB | P-D-D-P joint CSI | complete {n} x {n}", fontsize=12)
        fig.savefig(run/f"csi_snr_{value:02g}.png", dpi=160)
        plt.close(fig)
    write_json(run/"heatmap_scale.json", dict(reference_peak=peak, minimum_db=-80, maximum_db=vmax,
               error="complex estimated-reference before magnitude", amplitude_definition="20log10(abs(H)/shared_reference_peak)"))
    center = slice(max(n//2-32, 0), min(n//2+32, n))
    np.savez_compressed(run/"inspection_sample.npz", observations=h["observations"][:, index],
                        reference_csi_center=reference[:, center, center], estimated_csi_center=estimates[:, :, center, center],
                        actual_snr_db=h["actual_snr_db"][:, index], snr_db=snr, data_symbol_indices=[1, 2])


def write_report(r, run):
    run = Path(run)
    channel = html.escape(r["channel"])
    ep_parameters = r["ep_parameters"]
    prior = r["prior"]
    validation = json.loads((run/"validation.json").read_text(encoding="utf-8"))
    rows = "".join("<tr><td>"+name+"</td>"+"".join(f"<td>{e}/{b}<br>{v:.6g}</td>"
                   for e, b, v in zip(r["curves"][name]["errors"], r["curves"][name]["total_bits"],
                                      r["curves"][name]["ber"]))+"</tr>" for name in CURVES)
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{channel} MMSE / EP</title>
<style>body{{font:16px/1.65 system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#182c3d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dc;padding:8px}}pre{{white-space:pre-wrap;background:#f3f6f8;padding:15px}}</style>
<h1>{channel}：MMSE／EP</h1>
<p>{r['frame_count']} 帧，{r['data_symbol_count']} 个 D，SNR {r['snr_db']} dB；提交 {html.escape(r['provenance']['commit'])}。</p>
<p>两条 BER 曲线均使用接收机估计 CSI。MMSE 为单抽头均衡；EP 使用完整 ICI、公开导频消除和有色热噪声。参考 CSI 用于 NMSE 与热力图。</p>
<table><tr><th>算法</th>{''.join(f'<th>{s} dB</th>' for s in r['snr_db'])}</tr>{rows}</table>
<img src="ber_snr.png"><img src="nmse_snr.png">
<h2>整帧联合信道估计</h2>
<p>FFT1024、CP384，P1-D1-D2-P2。首尾P各1024行，D各选128导频及256保护空，共2816行一次求解公共DPSS系数，同时重建D1/D2。软反馈关闭，估计器不接触真实数据标签或参考CSI。</p>
<p>两信道固定实际半带宽{prior['bandwidth_hz']:.10g}Hz、尾比{prior['tail_ratio']:g}、Q={prior['count']}、L={prior['length']}、B={prior['block_size']}。051–055独立拟合本信道先验，056–060仅验证固定参数，001–010测试；不重新搜索。</p>
<p>模型误差协方差下限系数{r['csi_covariance_floor']['ratio']:g}，由公开P1/P2接收功率扣除热噪声并按P/D能量归一化。CP不足和支撑遗漏依配置保留为真实模型失配，数值重放和镜像检查继续执行。</p>
<p>EP 最多{ep_parameters['max_iterations']}轮、至少{ep_parameters['min_iterations']}轮，连续{ep_parameters['convergence_patience']}轮相对变化≤{ep_parameters['relative_tolerance']:g}时停止，阻尼{ep_parameters['damping']:g}、方差下限{ep_parameters['variance_floor']:g}。</p>
<p>每SNR每算法正式测试51200比特；NMSE先对全部D在线性域平均再转dB，图中只有一条平均曲线。噪声按D1/D2有效接收窗平均功率标定，CP不参与。</p>
<pre>{html.escape(json.dumps(dict(prior=prior, validation=validation, ep=r['ep_summary'], model_mismatch=r['model_mismatch']), indent=2, ensure_ascii=False))}</pre>
<h2>参考／估计／复数误差 CSI</h2>{''.join(f'<img src="csi_snr_{s:02g}.png">' for s in r['snr_db'])}
<p>热力图共用参考峰值和dB色标，下限−80dB；两行分别为D1/D2。</p></html>'''
    (run/"report.html").write_text(body, encoding="utf-8")
