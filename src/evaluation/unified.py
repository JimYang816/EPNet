"""One compensated-CSI report for the configured traditional/learned detectors."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from src.data.compact import CompactDataset
from src.evaluation import comparison
from src.evaluation.comparison_plots import publish_report
from src.evaluation.fingerprints import DataIdentity
from src.evaluation.result_cache import FileLock

LEARNED = ("udnet", "oamp_net2", "icr_mmse")
ORDER = ("Single-tap MMSE", "Full-matrix LMMSE", "MMSE-PIC", "UDNet", "OAMP-Net2", "ICR-MMSE-Net")


def _preflight(dataset_path, checkpoints, backend):
    if set(checkpoints)-set(LEARNED):
        raise ValueError("Unknown learned comparator checkpoint")
    for name, path in checkpoints.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}. Train it or select --baselines-only.")
    with CompactDataset(dataset_path) as ds:
        if ds.split != "test":
            raise ValueError("Unified comparison requires the test split")
        if not ds._config.get("smoke") and (ds.frame_count != 600 or len(ds._indices) != 512):
            raise ValueError("Unified comparison requires all 600 test frames and 512 data carriers")
        for name, path in checkpoints.items():
            if name == "udnet":
                from src.algorithms.udnet.checkpoint import load_model, validate_dataset
            elif name == "oamp_net2":
                from src.algorithms.oamp_net2.checkpoint import load_model, validate_dataset
            else:
                from src.algorithms.icr_mmse.checkpoint import load_model, validate_dataset
            model, payload = load_model(path, backend.device)
            validate_dataset(payload, ds)
            del model, payload
        return bool(ds._config.get("smoke"))


def evaluate(dataset_path, run, backend, *, checkpoints=None, batch_size=8, cache_dir=None,
             algorithm_parameters=None, force=(), input_view="compensated", bootstrap_samples=2000):
    if input_view != "compensated":
        raise ValueError("Unified experiment requires compensated CSI")
    checkpoints = dict(checkpoints or {})
    forced = {force} if isinstance(force, str) else set(force)
    smoke = _preflight(dataset_path, checkpoints, backend)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".unified_evaluation.lock"):
        guard = DataIdentity({name: Path(dataset_path).parent/name for name in ("test.h5", "model.h5")})
        baseline = comparison.evaluate(dataset_path, run/"baselines", backend, batch_size=batch_size,
            cache_dir=cache_dir, algorithm_parameters=algorithm_parameters, force=forced-{"icr_mmse"}, input_view=input_view)
        report = dict(baseline, schema="watermark-unified-comparison-v1", smoke=smoke, curves=dict(baseline["curves"]))
        components = {}
        if "udnet" in checkpoints:
            from src.evaluation.udnet import evaluate as evaluate_udnet
            components["udnet"] = evaluate_udnet(dataset_path, checkpoints["udnet"], run/"udnet", backend,
                batch_size=batch_size, cache_dir=cache_dir, algorithm_parameters=algorithm_parameters)
            report["curves"]["UDNet"] = components["udnet"]["curves"]["UDNet"]
            report["udnet"] = components["udnet"]["udnet"]
        if "oamp_net2" in checkpoints:
            from src.evaluation.oamp_net2 import evaluate as evaluate_oamp
            components["oamp_net2"] = evaluate_oamp(dataset_path, checkpoints["oamp_net2"], run/"oamp_net2", backend,
                batch_size=batch_size, cache_dir=cache_dir, algorithm_parameters=algorithm_parameters,
                force="all" in forced)
            report["curves"]["OAMP-Net2"] = components["oamp_net2"]["curves"]["OAMP-Net2"]
            report["oamp_net2"] = components["oamp_net2"]["oamp_net2"]
        if "icr_mmse" in checkpoints:
            from src.evaluation.icr_cache import evaluate as evaluate_icr
            components["icr_mmse"] = evaluate_icr(dataset_path, checkpoints["icr_mmse"], run/"icr_mmse", backend,
                batch_size=batch_size, cache_dir=cache_dir, bootstrap_samples=bootstrap_samples,
                force=bool(forced.intersection(("icr_mmse", "all"))))
            icr = components["icr_mmse"]
            report["curves"]["ICR-MMSE-Net"] = icr["curves"]["ICR-MMSE-Net"]
            report["nmse"] = dict(baseline["nmse"], corrected_mean_nmse_db=[r["mean_nmse_db"][-1] for r in icr["per_snr"]])
            report["icr_mmse"] = dict(checkpoint=icr["checkpoint"], sha256=icr["checkpoint_sha256"],
                model_config=icr["model_config"], comparison_report=str((run/"icr_mmse/metrics.json").resolve()),
                evaluation_cache=icr["evaluation_cache"])
        for name, component in components.items():
            component_data = {key: component["data_identity"].get(key) for key in guard.files}
            if (component_data != guard.files or component["snr_db"] != baseline["snr_db"]
                    or component.get("input_view") != "compensated"):
                raise RuntimeError(f"{name} comparison data/view/SNR identity mismatch")
            for label in set(component["curves"]).intersection(report["curves"]):
                if component["curves"][label]["total_bits"] != report["curves"][label]["total_bits"]:
                    raise RuntimeError("Comparators did not score the same number of test bits")
        if baseline["data_identity"] != guard.files:
            raise RuntimeError("Baseline data identity changed during unified evaluation")
        expected_bits = [baseline["bits_per_snr"]]*len(baseline["snr_db"])
        if any(curve["total_bits"] != expected_bits for curve in report["curves"].values()):
            raise RuntimeError("Unified comparison requires equal complete-test bit counts")
        if "icr_mmse" in components:
            initial = [row["mean_nmse_db"][0] for row in components["icr_mmse"]["per_snr"]]
            if not np.allclose(initial, report["nmse"]["mean_nmse_db"], rtol=0, atol=2e-5):
                raise ValueError("ICR input NMSE disagrees with the shared full-matrix CSI definition")
        report["curves"] = {name: report["curves"][name] for name in ORDER if name in report["curves"]}
        report["comparison_algorithms"] = list(report["curves"])
        report["checkpoint_paths"] = {name: str(Path(path).resolve()) for name, path in checkpoints.items()}
        report["nmse"]["semantics"] = "Shared fixed input CSI; only ICR-MMSE-Net has a corrected CSI curve"
        guard.assert_unchanged()
        publish_report(run, report)
        _write_html(run, report)
        return report


def _write_html(run, report):
    headings = "".join(f"<th>{s:g} dB</th>" for s in report["snr_db"])
    rows = "".join("<tr><th>"+html.escape(name)+"</th>"+"".join(f"<td>{v:.6g}</td>" for v in curve["ber"])+"</tr>"
                   for name, curve in report["curves"].items())
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{report['channel']} 均衡算法对比</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dc;padding:8px}}</style>
<h1>{report['channel']}：统一均衡评测</h1><p>固定测试划分、compensated CSI、D1/D2，共 {report['bits_per_snr']:,} 比特 / SNR。模型选择仅使用训练集和验证集。</p>
<table><tr><th>算法 BER</th>{headings}</tr>{rows}</table><img src="ber_snr.png"><img src="nmse_snr.png">
<p>零误码用空心三角在 0.5 / 测试比特数处显示。输入 CSI 的 NMSE 为逐帧、逐 D 全矩阵线性相对误差先平均再转 dB；ICR 单独报告修正后的 CSI。</p>
<p>检查点、数据、代码与缓存来源见 metrics.json。</p></html>'''
    (run/"report.html").write_text(body, encoding="utf-8")
