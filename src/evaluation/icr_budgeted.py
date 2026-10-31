"""Frozen comparisons and synchronized timing for budgeted MMSE unfolding."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.algorithms.icr_mmse.checkpoint import load_model, validate_dataset, INFERENCE_CODE
from src.algorithms.icr_mmse.model import ICRMMSEnet, csi_nmse
from src.algorithms.icr_mmse.lmmse import FullMatrixLMMSELayer
from src.algorithms.icr_mmse.inference import prepare_frozen_inference
from src.algorithms.icr_mmse.legacy_20261030 import ICRMMSEnet as LegacyICRMMSEnet
from src.algorithms.icr_mmse.legacy_lmmse_20261030 import FullMatrixLMMSELayer as LegacyLMMSELayer
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.data.icr_dense_training import dense_input_batches
from src.data.icr_mmse import supervision
from src.data.icr_training import ReferenceCache
from src.evaluation.fingerprints import DataIdentity, source_fingerprint, numerical_environment, generation_origin
from src.evaluation.icr_mmse import RecordMetrics, _initial_nmse, _record_disjoint, record_interval

LEGACY_CODE = ("src.algorithms.icr_mmse.legacy_20261030", "src.algorithms.icr_mmse.legacy_lmmse_20261030")
EVALUATION_CODE = INFERENCE_CODE+("src.evaluation.icr_budgeted", "src.algorithms.icr_mmse.inference")+LEGACY_CODE
HISTORICAL_BER_ATOL = 1e-5


def reproduction_audit(expected, actual, bits):
    """Report tiny floating-point replay differences; never hide a large drift."""
    expected, actual, bits = (np.asarray(x, dtype=np.int64) for x in (expected, actual, bits))
    if expected.shape != actual.shape or bits.shape != actual.shape or np.any(bits <= 0):
        raise ValueError("Historical replay count coverage differs")
    difference = actual-expected
    absolute_ber_difference = np.abs(difference)/bits
    return dict(expected_errors=expected.tolist(), actual_errors=actual.tolist(),
                bit_error_difference=difference.tolist(), absolute_ber_difference=absolute_ber_difference.tolist(),
                absolute_ber_tolerance=HISTORICAL_BER_ATOL, exact=bool(np.all(difference == 0)),
                within_tolerance=bool(np.all(absolute_ber_difference <= HISTORICAL_BER_ATOL)))


def legacy_model(config):
    """Use the experiment's exact forward path, without later log-state work."""
    config = {k: v for k, v in config.items()
              if k not in ("step_mode", "initial_soft_damping", "initial_csi_loading")}
    model = LegacyICRMMSEnet(**config)
    if config["detector_mode"] == "lmmse":
        model.equalizers = torch.nn.ModuleList(LegacyLMMSELayer() for _ in model.equalizers)
    return model


class LoadedLMMSEControl(FullMatrixLMMSELayer):
    """Diagnostic: give exact LMMSE the same learned CSI-error regularizer.

    This separates the benefit of noise/loading calibration from the benefit
    of repeated nonlinear QPSK updates. It is never used to train or select the
    unfolding checkpoint.
    """

    def __init__(self, loading_log):
        super().__init__()
        self.register_buffer("loading", torch.nn.functional.softplus(loading_log.detach()).clone())

    def forward(self, channel, observations, known, mean, variance, data_mask, pilot_mask,
                null_mask, positions, rf_variance, covariance_factor, constellation, **kwargs):
        weights = kwargs.get("column_weights")
        if weights is None:
            weights = kwargs["covariance_diagonal"].reciprocal()
        diagonal = channel.column_energy(weights, base_energy=kwargs.get("base_column_energy")).clamp_min(1e-20)
        energy = (diagonal*data_mask).sum(-1)/data_mask.sum().clamp_min(1)
        return super().forward(channel, observations, known, mean, variance, data_mask, pilot_mask,
            null_mask, positions, rf_variance+self.loading*energy, covariance_factor, constellation, **kwargs)


def paired(candidate, reference):
    if [r["snr_db"] for r in candidate] != [r["snr_db"] for r in reference]:
        raise ValueError("Paired SNR coverage differs")
    total = defaultdict(lambda: [0, 0])
    per_snr = []
    for a, b in zip(candidate, reference):
        ar = {r["record"]: r for r in a["by_record"]}
        br = {r["record"]: r for r in b["by_record"]}
        if set(ar) != set(br):
            raise ValueError("Paired record coverage differs")
        diff, bits = [], []
        for record in sorted(ar):
            if ar[record]["bits"] != br[record]["bits"]:
                raise ValueError("Paired bit counts differ")
            d = ar[record]["bit_errors"][-1] - br[record]["bit_errors"][-1]
            n = ar[record]["bits"]
            diff.append(d)
            bits.append(n)
            total[record][0] += d
            total[record][1] += n
        per_snr.append(dict(snr_db=a["snr_db"], **record_interval(diff, bits)))
    return dict(per_snr=per_snr, overall=record_interval(
        [v[0] for v in total.values()], [v[1] for v in total.values()]))


def _benchmark(models, inputs, backend, repeats=11):
    """Warm, rotated-order timing. Every call includes all detector/CSI layers."""
    samples = {name: [] for name in models}
    names = list(models)
    for _ in range(3):
        for model in models.values():
            model(**inputs, return_diagnostics=False)
    backend.synchronize()
    for r in range(repeats):
        for name in names[r % len(names):] + names[:r % len(names)]:
            backend.synchronize()
            tick = time.perf_counter()
            models[name](**inputs, return_diagnostics=False)
            backend.synchronize()
            samples[name].append(time.perf_counter()-tick)
    batch = len(inputs["observations"])
    return {name: dict(median_seconds_per_block=float(np.median(v))/batch,
                      min_seconds_per_block=min(v)/batch, max_seconds_per_block=max(v)/batch,
                      seconds_per_batch=v) for name, v in samples.items()}


def _historical_reference(checkpoint, dataset, device):
    """Explicit frozen baseline import, authenticated against its original run.

    Never relabel the historical checkpoint with a current source fingerprint.
    Its digest and data identity must match the completed A-D experiment, and
    full test BER is checked against that experiment after prediction.
    """
    checkpoint = Path(checkpoint)
    quartet = json.loads((checkpoint.parents[1]/"quartet.json").read_text())
    variant = checkpoint.parent.name
    digest = file_sha256(checkpoint)
    if not quartet.get("complete") or quartet["checkpoints"][variant]["sha256"] != digest:
        raise ValueError("Historical baseline does not match the frozen quartet")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_dataset(payload, dataset)
    if payload.get("input_view") != "compensated" or payload.get("smoke"):
        raise ValueError("Historical baseline requires accepted compensated data")
    for name in ("train.h5", "validation.h5", "model.h5"):
        if payload["identity"]["data"][name]["sha256"] != file_sha256(dataset.path.parent/name):
            raise ValueError("Historical training data differ")
    legacy_source = source_fingerprint(LEGACY_CODE)
    for archived, original in zip(LEGACY_CODE, ("src.algorithms.icr_mmse.model", "src.algorithms.icr_mmse.lmmse")):
        if legacy_source[archived] != payload["inference_code"][original]:
            raise ValueError("Archived inference does not match the historical experiment source")
    shared = {k: v for k, v in payload["inference_code"].items() if k not in
              ("src.algorithms.icr_mmse.model", "src.algorithms.icr_mmse.lmmse", "src.algorithms.icr_mmse.checkpoint")}
    if source_fingerprint(tuple(shared)) != shared:
        raise ValueError("Historical shared data/operator source differs")
    model = legacy_model(payload["model_config"]).to(device)
    if model.config["detector_mode"] not in ("unfolded", "lmmse"):
        raise ValueError("Historical baseline must use the original detector")
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, dict(path=str(checkpoint), sha256=digest, original_inference_code=payload["inference_code"],
                       archived_source=legacy_source, shared_source_verified=True,
                       expected_test_bits=[r["total_bits"] for r in quartet["rows"]],
                       expected_test_errors=[r["variants"][variant]["ber"]*r["total_bits"]
                                             for r in quartet["rows"]])


def compare(dataset_path, checkpoint, run, backend, *, split="validation", batch_size=2,
            historical=None, cache_mib=8192):
    run, checkpoint = Path(run), Path(checkpoint)
    if (run/"comparison.json").exists():
        raise FileExistsError("Use a new frozen comparison output")
    run.mkdir(parents=True, exist_ok=True)
    model, payload = load_model(checkpoint, backend.device)
    if model.config["detector_mode"] != "budgeted":
        raise ValueError("Comparison requires a budgeted unfolding checkpoint")
    if payload.get("smoke"):
        raise ValueError("Formal comparison cannot use a smoke checkpoint")
    checksum = file_sha256(checkpoint)
    torch.manual_seed(payload["seed"])
    initial = ICRMMSEnet(**model.config).to(backend.device)
    original_config = dict(model.config, detector_mode="unfolded", step_mode="fixed", initial_soft_damping=None)
    torch.manual_seed(payload["seed"])
    original = legacy_model(original_config).to(backend.device)
    lmmse_config = dict(model.config, detector_mode="lmmse")
    lmmse = legacy_model(lmmse_config).to(backend.device)
    loaded = legacy_model(lmmse_config).to(backend.device)
    loaded.equalizers = torch.nn.ModuleList(LoadedLMMSEControl(
        model.equalizers[-1 if model.config["correction_mode"] == "fixed" else i].csi_loading_log)
        for i in range(model.config["layers"]))
    # Matched-weights controls isolate detector changes; the historical trained
    # ICR+LMMSE reference can additionally be supplied for the stronger comparison.
    if model.config["correction_mode"] != "fixed":
        correction = {k: v for k, v in model.state_dict().items()
                      if k.startswith("corrector.") or k in ("eta_logits", "constellation")}
        for other in (initial, original, lmmse, loaded):
            other.load_state_dict(dict(other.state_dict(), **correction))
    models = dict(lmmse=lmmse, loaded_lmmse=loaded, original=original, initialized=initial, repaired=model)
    provenance = {}
    accumulators, seconds = {}, defaultdict(float)
    preparation, blocks = 0., 0
    cache = ReferenceCache(cache_mib*2**20)
    with CompactDataset(dataset_path, input_view="compensated") as dataset, torch.inference_mode():
        if split not in ("validation", "test") or dataset.split != split:
            raise ValueError("Explicit validation/test split identity required")
        validate_dataset(payload, dataset)
        _record_disjoint(dataset)
        guard = DataIdentity({n: dataset.path.parent/n for n in (dataset.path.name, "model.h5", "train.h5", "validation.h5")})
        for name in ("train.h5", "validation.h5", "model.h5"):
            if payload.get("identity", {}).get("data", {}).get(name) != guard.files[name]:
                raise ValueError(f"Candidate checkpoint training data differ: {name}")
        for name, path in (historical or {}).items():
            if name in models:
                raise ValueError("Duplicate comparison label")
            models[name], provenance[name] = _historical_reference(path, dataset, backend.device)
        for item in models.values():
            item.eval().requires_grad_(False)
            prepare_frozen_inference(item)
        iterator = iter(dense_input_batches(dataset, batch_size=batch_size,
                        precision=model.config["precision"], reference_cache=cache))
        bases, shared, benchmark = {}, {}, None
        while True:
            backend.synchronize()
            tick = time.perf_counter()
            try:
                cpu_inputs, keys = next(iterator)
            except StopIteration:
                break
            source = cpu_inputs["initial_channel"]
            if id(source.basis) not in bases:
                bases[id(source.basis)] = source.basis.to(backend.device)
            reference = source.reference.to(backend.device)
            inputs = dict(initial_channel=source.dense(backend.device,
                basis=bases[id(source.basis)], reference=reference))
            for name, value in cpu_inputs.items():
                if name == "initial_channel":
                    continue
                if name in ("data_mask", "pilot_mask", "null_mask", "carrier_positions", "unit_covariance"):
                    if name not in shared:
                        shared[name] = value.to(backend.device)
                    inputs[name] = shared[name]
                else:
                    inputs[name] = value.to(backend.device)
            backend.synchronize()
            preparation += time.perf_counter()-tick
            if benchmark is None:
                benchmark = _benchmark(models, inputs, backend)
            predictions, nmse_values = {}, {}
            for name, item in models.items():
                tick = time.perf_counter()
                output = item(**inputs, return_diagnostics=False)
                backend.synchronize()
                seconds[name] += time.perf_counter()-tick
                predictions[name] = output.symbols[..., inputs["data_mask"]]
                # CSI scoring is outside the prediction timer, never a model input.
                before = _initial_nmse(inputs["initial_channel"], reference)
                after = csi_nmse(output, reference)
                nmse_values[name] = torch.cat((before[:, None], after), -1).cpu().numpy()
                del output
            # First bit-label access occurs after every model has predicted.
            targets = supervision(dataset, keys, device=backend.device, include_reference=False,
                                  precision=model.config["precision"])
            expected = targets["symbols"][:, None]
            records = targets["records"].cpu().tolist()
            for name, prediction in predictions.items():
                errors = (((prediction.real < 0) != (expected.real < 0)).sum(-1)
                          + ((prediction.imag < 0) != (expected.imag < 0)).sum(-1)).cpu().numpy()
                accumulator = accumulators.setdefault(name, RecordMetrics(errors.shape[-1]))
                for i, ((si, _, _), record) in enumerate(zip(keys, records)):
                    accumulator.add(si, record, errors[i], 2*int(inputs["data_mask"].sum()), nmse_values[name][i])
            blocks += len(keys)
            if blocks % (20*batch_size) == 0:
                write_json(run/"progress.json", dict(split=split, blocks=blocks,
                    total_blocks=dataset.frame_count*2*len(dataset.snr_db)))
        reports, historical_reproduction = {}, {}
        for name, accumulator in accumulators.items():
            rows = accumulator.report(dataset.snr_db)
            if name in provenance and split == "test":
                expected_errors = np.rint(provenance[name]["expected_test_errors"]).astype(np.int64)
                actual = np.array([r["bit_errors"][-1] for r in rows])
                bits = [r["total_bits"] for r in rows]
                if bits != provenance[name]["expected_test_bits"]:
                    raise ValueError("Historical test bit coverage differs")
                audit = reproduction_audit(expected_errors, actual, bits)
                historical_reproduction[name] = audit
                if not audit["within_tolerance"]:
                    write_json(run/"historical_reproduction_failure.json", dict(model=name, **audit))
                    raise RuntimeError(f"Historical {name} exceeds fixed numerical replay tolerance: {audit['bit_error_difference']}")
            reports[name] = dict(model_config=models[name].config, per_snr=rows,
                mean_ber=sum(r["bit_errors"][-1] for r in rows)/sum(r["total_bits"] for r in rows),
                detector_parameters=sum(p.numel() for p in models[name].equalizers.parameters()),
                stored_parameters=sum(p.numel() for p in models[name].parameters()),
                seconds_per_block=(preparation+seconds[name])/blocks,
                model_seconds_per_block=seconds[name]/blocks, benchmark=benchmark[name])
        report = dict(complete=True, input_view="compensated", split=split, channel=dataset.channel,
            origin=generation_origin(run, backend, batch_size),
            environment=dict(numerical_environment(backend), model_precision=model.config["precision"],
                             precision=model.config["precision"], torch_threads=torch.get_num_threads(),
                             deterministic=torch.are_deterministic_algorithms_enabled()),
            test_access=split == "test", data_identity=guard.files, checkpoint_sha256=checksum,
            source=source_fingerprint(EVALUATION_CODE), models=reports,
            deployment="Budgeted models cache frozen scalar activations and window geometry; exactly zero classifier heads are omitted. Training/checkpoint equations are unchanged.",
            historical_provenance=provenance, preparation_seconds=preparation, blocks=blocks,
            historical_reproduction=historical_reproduction,
            batch_size=batch_size,
            controls="initialized resets only the detector; original replaces it with the legacy initialized "
                     "detector; lmmse replaces it with exact LMMSE; each retains the repaired model's CSI "
                     "corrector weights, when enabled; loaded_lmmse uses the repaired detector's learned "
                     "CSI loading (last layer for fixed CSI); historical references retain their own trained weights",
            timing="Shared receiver input preparation charged once to EACH model; synchronized model timers; "
                   "all CSI and detector layers; labels, metrics, warmup excluded",
            comparisons={name: paired(reports["repaired"]["per_snr"], row["per_snr"])
                         for name, row in reports.items() if name != "repaired"})
        guard.assert_unchanged()
        if file_sha256(checkpoint) != checksum:
            raise RuntimeError("Frozen checkpoint changed during comparison")
        write_json(run/"comparison.json", report)
    plot(report, run)
    return report


def plot(report, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ("ber", "mean_nmse_db"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, item in report["models"].items():
            rows = item["per_snr"]
            ax.plot([r["snr_db"] for r in rows], [r[metric][-1] for r in rows], "o-", label=name)
        if metric == "ber":
            ax.set_yscale("log")
        ax.set(xlabel="SNR (dB)", ylabel="BER" if metric == "ber" else "Mean full-matrix NMSE (dB)",
               title=f'{report["channel"]}: {report["split"]}, compensated CSI')
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(run)/("ber_snr.png" if metric == "ber" else "nmse_snr.png"), dpi=180)
        plt.close(fig)
