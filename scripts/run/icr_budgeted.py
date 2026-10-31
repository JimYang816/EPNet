"""Validation-only initialization probe for equal-compute MMSE unfolding."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.algorithms.icr_mmse.checkpoint import load_model, validate_dataset
from src.common.experiment import write_json, file_sha256
from src.data.compact import CompactDataset
from src.data.icr_training import ReferenceCache, training_batches
from src.evaluation.fingerprints import DataIdentity
from src.evaluation.icr_budgeted import LoadedLMMSEControl
from src.training.icr_mmse import run_epoch


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ("ber", "corrected_nmse"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, row in report["models"].items():
            values = row["validation"]["per_snr"]
            y = [x[metric] for x in values]
            if metric == "corrected_nmse":
                y = 10*np.log10(y)
            ax.plot([x["snr_db"] for x in values], y, marker="o", label=name)
        if metric == "ber":
            ax.set_yscale("log")
        ax.set(xlabel="SNR (dB)", ylabel="BER" if metric == "ber" else "Mean full-matrix NMSE (dB)",
               title=f'{report["channel"]}: validation only')
        ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output/("ber_snr.png" if metric == "ber" else "nmse_snr.png"), dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, help="Probe a frozen candidate versus LMMSE with identical CSI-corrector weights")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new probe output directory")
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads), CompactDataset(args.dataset/"validation.h5") as dataset:
        if dataset.split != "validation" or dataset.input_view != "compensated":
            raise ValueError("Probe requires compensated validation data")
        guard = DataIdentity({name: args.dataset/name for name in ("validation.h5", "model.h5")})
        frames = np.linspace(0, dataset.frame_count-1, min(args.frames, dataset.frame_count), dtype=int).tolist()
        common = dict(layers=args.layers, bandwidth=16, correction_mode="fixed", noise_mode="diagonal")
        options = {"LMMSE": dict(common, detector_mode="lmmse"),
                   "original": dict(common, detector_mode="unfolded")}
        for loading in (.001, .01, .03, .1):
            for damping in (.5, .8):
                options[f"loading_{loading}_damping_{damping}"] = dict(common,
                    detector_mode="budgeted", initial_csi_loading=loading, initial_soft_damping=damping)
        report = dict(channel=dataset.channel, split="validation", test_access=False, input_view="compensated",
                      frames=frames, data_identity=guard.files, models={})
        prepared = {}
        if args.checkpoint:
            repaired, payload = load_model(args.checkpoint, args.device)
            validate_dataset(payload, dataset)
            if any(payload["identity"]["data"][k] != value for k, value in guard.files.items()):
                raise ValueError("Checkpoint validation data identity differs")
            lmmse = ICRMMSEnet(**dict(repaired.config, detector_mode="lmmse")).to(args.device)
            loaded = ICRMMSEnet(**dict(repaired.config, detector_mode="lmmse")).to(args.device)
            loaded.equalizers = torch.nn.ModuleList(LoadedLMMSEControl(
                repaired.equalizers[-1 if repaired.config["correction_mode"] == "fixed" else i].csi_loading_log)
                for i in range(repaired.config["layers"]))
            correction = {k: v for k, v in repaired.state_dict().items()
                          if k.startswith("corrector.") or k in ("eta_logits", "constellation")}
            lmmse.load_state_dict(dict(lmmse.state_dict(), **correction))
            loaded.load_state_dict(dict(loaded.state_dict(), **correction))
            prepared = dict(repaired=repaired, lmmse=lmmse, loaded_lmmse=loaded)
            options = {name: item.config for name, item in prepared.items()}
            report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
                          control="Same CSI corrector weights; detector-specific soft feedback")
        cache = ReferenceCache(4096*2**20)
        for name, config in options.items():
            torch.manual_seed(20261028)
            model = prepared[name] if name in prepared else ICRMMSEnet(**config).to(args.device)
            tick = time.perf_counter()
            stage = "A" if model.config["correction_mode"] == "fixed" else "C"
            metrics = run_epoch(model, dataset, args.device, stage=stage, batch_size=8,
                frame_indices=frames, operator_backend="dense_device", prefetch_batches=2,
                reference_cache=cache, loss_options=dict(objective="bit_ce", final_weight=.9))
            report["models"][name] = dict(config=model.config, validation=metrics,
                                           scoring_seconds=time.perf_counter()-tick)
            write_json(args.output/"probe.json", report)
            logging.info("%s mean BER %.8f, per SNR %s", name, metrics["ber"],
                         [round(r["ber"], 7) for r in metrics["per_snr"]])
        guard.assert_unchanged()
        if args.checkpoint and file_sha256(args.checkpoint) != report["checkpoint_sha256"]:
            raise RuntimeError("Probe checkpoint changed")
        report["complete"] = True
        write_json(args.output/"probe.json", report)
        plot(report, args.output)


if __name__ == "__main__":
    main()
