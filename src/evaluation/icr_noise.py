"""Paired full/diagonal noise ablation with identical learned weights and inputs."""
from __future__ import annotations

from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.algorithms.icr_mmse.checkpoint import INFERENCE_CODE, load_model, validate_dataset
from src.algorithms.icr_mmse.model import ICRMMSEnet, csi_nmse
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.data.icr_mmse import input_batches, supervision
from src.evaluation.fingerprints import DataIdentity, numerical_environment, source_fingerprint
from src.evaluation.icr_mmse import RecordMetrics, _initial_nmse, _record_disjoint
from src.evaluation.result_cache import FileLock


class _FrameSubset:
    """Read-only input view; exported batch keys are mapped back before scoring."""
    def __init__(self, source, indices):
        self.source, self.indices = source, indices
        self.frame_count = len(indices)

    def __getattr__(self, name):
        return getattr(self.source, name)

    def model_input(self, snr_index, frame_index):
        return self.source.model_input(snr_index, self.indices[frame_index])


def _sample_batches(dataset, indices, **options):
    yield from input_batches(dataset, frame_indices=indices, **options)


def _score(accumulator, output, inputs, keys, target):
    mask = inputs["data_mask"]
    predicted, expected = output.symbols[..., mask], target["symbols"][:, None, :]
    errors = ((predicted.real < 0) != (expected.real < 0)).sum(-1)
    errors += ((predicted.imag < 0) != (expected.imag < 0)).sum(-1)
    nmse = torch.cat((_initial_nmse(inputs["initial_channel"], target["reference_channel"])[:, None],
                      csi_nmse(output, target["reference_channel"])), -1)
    for key, record, row_errors, row_nmse in zip(keys, target["records"].cpu().tolist(),
                                                errors.cpu().numpy(), nmse.cpu().numpy()):
        accumulator.add(key[0], record, row_errors, 2 * int(mask.sum()), row_nmse)


def compare_noise(dataset_path, checkpoint, run, backend, *, batch_size=2, frame_limit=16,
                  timing_repeats=5, bootstrap_samples=200, frame_selection="evenly_spaced"):
    """Change only C^-1 actions; score after timing, without fitting on test data.

    Both modes reuse each reconstructed raw batch. Timing alternates mode order
    and excludes labels/scoring; end-to-end estimates add the same measured
    input preparation cost to each mode's median repeated inference time.
    The first batch has a separate cold forward to populate runtime caches.
    """
    if (any(isinstance(x, bool) or not isinstance(x, int) or x < 1
            for x in (batch_size, timing_repeats, bootstrap_samples))
            or (frame_limit is not None and (isinstance(frame_limit, bool)
                or not isinstance(frame_limit, int) or frame_limit < 1))
            or frame_selection not in ("first", "evenly_spaced")):
        raise ValueError("Invalid noise comparison controls")
    run, checkpoint = Path(run), Path(checkpoint)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run / ".noise_comparison.lock"):
        checkpoint_hash = file_sha256(checkpoint)
        original, payload = load_model(checkpoint, backend.device)
        models, setup, cold = {}, {}, {}
        for mode in ("full", "diagonal"):
            backend.synchronize()
            tick = time.perf_counter()
            model = ICRMMSEnet(**{**original.config, "noise_mode": mode}).to(backend.device)
            model.load_state_dict(original.state_dict(), strict=True)
            models[mode] = model.eval().requires_grad_(False)
            backend.synchronize()
            setup[mode] = time.perf_counter() - tick
        del original
        precision = models["full"].config["precision"]
        totals = {mode: RecordMetrics(model.config["layers"]) for mode, model in models.items()}
        times = {mode: np.zeros(timing_repeats) for mode in models}
        preparation, blocks, batches = 0., 0, 0
        maximum_probability_difference = 0.
        with CompactDataset(dataset_path, input_view="compensated") as dataset, torch.inference_mode():
            validate_dataset(payload, dataset)
            if dataset.split != "test":
                raise ValueError("Noise comparison requires the independent test split")
            _record_disjoint(dataset)
            identity = DataIdentity({name: dataset.path.parent / name
                                     for name in (dataset.path.name, "model.h5")})
            count = min(dataset.frame_count, frame_limit or dataset.frame_count)
            indices = (np.linspace(0, dataset.frame_count - 1, count, dtype=int).tolist()
                       if frame_selection == "evenly_spaced" else list(range(count)))
            iterator = iter(_sample_batches(dataset, indices, device=backend.device,
                                            batch_size=batch_size, precision=precision))
            while True:
                backend.synchronize()
                tick = time.perf_counter()
                try:
                    inputs, keys = next(iterator)
                except StopIteration:
                    break
                backend.synchronize()
                preparation += time.perf_counter() - tick
                if batches == 0:
                    for mode, model in models.items():
                        backend.synchronize()
                        tick = time.perf_counter()
                        warmup = model(**inputs)
                        backend.synchronize()
                        cold[mode] = time.perf_counter() - tick
                        del warmup
                predictions = {}
                for repeat in range(timing_repeats):
                    order = ("full", "diagonal") if (batches + repeat) % 2 == 0 else ("diagonal", "full")
                    for mode in order:
                        backend.synchronize()
                        tick = time.perf_counter()
                        prediction = models[mode](**inputs)
                        backend.synchronize()
                        times[mode][repeat] += time.perf_counter() - tick
                        if repeat == 0:
                            predictions[mode] = prediction
                        del prediction
                maximum_probability_difference = max(maximum_probability_difference,
                    float((predictions["full"].probabilities - predictions["diagonal"].probabilities).abs().max()))
                # References and labels are first opened after both inference modes.
                target = supervision(dataset, keys, device=backend.device, precision=precision)
                for mode in models:
                    _score(totals[mode], predictions[mode], inputs, keys, target)
                blocks += len(keys)
                batches += 1
                del predictions, inputs, target
            if not blocks:
                raise ValueError("Noise comparison requires at least one batch")
            modes = {}
            for mode in models:
                rows = totals[mode].report(dataset.snr_db, bootstrap_samples=bootstrap_samples)
                total_bits = sum(row["total_bits"] for row in rows)
                nmse = sum(row["mean_nmse_linear"][-1] * row["blocks"] for row in rows) / blocks
                inference = float(statistics.median(times[mode]))
                modes[mode] = dict(model_config=models[mode].config, per_snr=rows,
                    aggregate=dict(ber=sum(row["bit_errors"][-1] for row in rows) / total_bits,
                        mean_nmse_linear=nmse, mean_nmse_db=float(10 * np.log10(max(nmse, 1e-300))),
                        total_bits=total_bits, bit_errors=sum(row["bit_errors"][-1] for row in rows)),
                    performance=dict(inference_seconds_by_repeat=times[mode].tolist(),
                        median_inference_seconds=inference, inference_seconds_per_block=inference / blocks,
                        preprocessing_and_inference_seconds=preparation + inference,
                        end_to_end_seconds_per_block=(preparation + inference) / blocks,
                        model_construction_seconds=setup[mode], cold_first_batch_seconds=cold[mode]))
            full, diagonal = modes["full"], modes["diagonal"]
            report = dict(schema="icr-mmse-paired-noise-comparison-v1", complete=True, smoke=True,
                channel=dataset.channel, split=dataset.split, input_view="compensated", precision=precision,
                checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                trained_noise_mode=payload["model_config"].get("noise_mode", "full"),
                dataset=str(dataset.path.resolve()), data_identity=identity.files,
                code=source_fingerprint(INFERENCE_CODE + ("src.evaluation.icr_noise", "src.evaluation.icr_mmse")),
                numerical_environment=dict(numerical_environment(backend), torch=str(torch.__version__), precision=precision),
                sample_selection=frame_selection, selected_frame_indices=indices,
                frame_limit=frame_limit, snr_db=dataset.snr_db.tolist(), blocks=blocks, batches=batches,
                timing_repeats=timing_repeats, shared_input_preparation_seconds=preparation,
                timing_definition="warm inference median over repeated passes, alternating mode order; identical measured raw reconstruction/input preparation added to both modes; excludes checkpoint loading, construction, cold first-batch warmup, labels and scoring",
                comparison="same learned weights and raw observations; only noise inverse actions differ; no per-mode retraining",
                maximum_probability_difference=maximum_probability_difference, modes=modes,
                differences=dict(ber=diagonal["aggregate"]["ber"] - full["aggregate"]["ber"],
                    mean_nmse_db=diagonal["aggregate"]["mean_nmse_db"] - full["aggregate"]["mean_nmse_db"],
                    inference_speedup=full["performance"]["median_inference_seconds"] / diagonal["performance"]["median_inference_seconds"],
                    end_to_end_speedup=full["performance"]["preprocessing_and_inference_seconds"] / diagonal["performance"]["preprocessing_and_inference_seconds"]))
            identity.assert_unchanged()
            if file_sha256(checkpoint) != checkpoint_hash:
                raise RuntimeError("Checkpoint changed during paired comparison")
        write_json(run / "comparison.json", report)
        plot_comparison(report, run)
        return report


def plot_comparison(report, run):
    run = Path(run)
    snr = report["snr_db"]
    for metric, ylabel, filename in (("ber", "BER", "ber_snr.png"),
                                      ("mean_nmse_db", "Mean full-matrix CSI NMSE (dB)", "nmse_snr.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for mode, result in report["modes"].items():
            values = [row[metric][-1] for row in result["per_snr"]]
            if metric == "ber":
                ax.semilogy(snr, np.maximum(values, 1e-8), "o-", label=mode)
            else:
                ax.plot(snr, values, "o-", label=mode)
        if metric == "mean_nmse_db":
            ax.plot(snr, [row[metric][0] for row in report["modes"]["full"]["per_snr"]], "k--", label="initial CSI")
        ax.set(xlabel="SNR (dB)", ylabel=ylabel, title=report["channel"] + " noise approximation (small sample)")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    positions = np.arange(2)
    for offset, (key, label) in enumerate((("inference_seconds_per_block", "inference"),
                                          ("end_to_end_seconds_per_block", "with raw input preparation"))):
        values = [1000 * mode["performance"][key] for mode in report["modes"].values()]
        ax.bar(positions + (offset - .5) * .35, values, .35, label=label)
    ax.set(xticks=positions, xticklabels=list(report["modes"]), ylabel="Milliseconds per D block",
           title=report["channel"] + " paired runtime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "runtime.png", dpi=180)
    plt.close(fig)
