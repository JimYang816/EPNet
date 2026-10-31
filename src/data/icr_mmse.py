"""Selected-view receiver batches and separately requested CSI/symbol supervision."""
from __future__ import annotations

import numpy as np
import torch

from src.algorithms.icr_mmse.dpss import DPSSBasis, DPSSChannel

def precision_dtypes(precision):
    """Use matched real/complex arithmetic; double precision is an explicit check."""
    if precision not in ("complex64", "complex128"):
        raise ValueError("ICR precision must be complex64 or complex128")
    return ((torch.complex64, torch.float32) if precision == "complex64"
            else (torch.complex128, torch.float64))


def input_batches(dataset, *, device="cpu", batch_size=4, frame_limit=None, snr_indices=None,
                  precision="complex64", operator_backend="dense", frame_indices=None):
    """Yield (model inputs, sample keys), without reading any supervision fields.

    A key is (SNR index, physical frame index, D-block index). D1 and D2
    share the same model; FFT positions include pilots and nulls uncompressed.
    """
    if dataset.input_view not in ("raw", "compensated"):
        raise ValueError("Unknown ICR CSI input view")
    if operator_backend not in ("dense", "dpss"):
        raise ValueError("Unknown ICR operator backend")
    if dataset.input_view == "compensated" and operator_backend == "dpss":
        raise ValueError("Compensated CSI includes full reference support; use dense or dense_device")
    complex_dtype, real_dtype = precision_dtypes(precision)
    if (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1
            or (frame_limit is not None and (not isinstance(frame_limit, int)
                or isinstance(frame_limit, bool) or frame_limit < 1))):
        raise ValueError("Invalid ICR batch controls")
    count = dataset.frame_count if frame_limit is None else min(frame_limit, dataset.frame_count)
    indices = list(range(count)) if frame_indices is None else list(frame_indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(isinstance(fi, bool) or not isinstance(fi, (int, np.integer))
                   or not 0 <= fi < dataset.frame_count for fi in indices)):
        raise ValueError("Frame indices must be distinct valid physical frame indices")
    selected = list(range(len(dataset.snr_db))) if snr_indices is None else list(snr_indices)
    if any(not 0 <= si < len(dataset.snr_db) for si in selected):
        raise IndexError("Invalid SNR index")
    numpy_complex = np.complex64 if precision == "complex64" else np.complex128
    numpy_real = np.float32 if precision == "complex64" else np.float64
    n = dataset._config["nfft"]
    masks = {name: torch.as_tensor(value, dtype=torch.bool, device=device)
             for name, value in dataset._masks.items()}
    data_mask, pilot_mask = masks["data"], masks["pilot"]
    null_mask = ~(data_mask | pilot_mask)
    covariance = torch.as_tensor(np.asarray(dataset._shared["unit_rf_noise_covariance"][:], dtype=numpy_complex),
                                 dtype=complex_dtype, device=device)
    known = [torch.as_tensor(np.asarray(dataset._known[d+1], dtype=numpy_complex),
                             dtype=complex_dtype, device=device) for d in range(2)]
    positions = torch.arange(n, device=device)
    bases = None
    if operator_backend == "dpss":
        # Use the exact saved basis and FFT-window sampling, never refit DPSS.
        sampled = dataset._model.sampled_basis
        bases = [DPSSBasis(torch.as_tensor(sampled[(d+1)*n:(d+2)*n],
                                           dtype=complex_dtype, device=device)) for d in range(2)]
    for si in selected:
        for d in range(2):
            for start in range(0, len(indices), batch_size):
                frames = indices[start:start+batch_size]
                # Reconstruct on CPU, then transfer only the requested precision.
                # No fixed-H pilot cancellation is needed: the model recomputes it
                # after every CSI update. Avoid a second complex128 CUDA channel.
                channels = (np.empty((len(frames), n, n), dtype=numpy_complex) if bases is None else
                            np.empty((len(frames), dataset._model.count, dataset._model.length), dtype=numpy_complex))
                observations = np.empty((len(frames), n), dtype=numpy_complex)
                variances = np.empty(len(frames), dtype=numpy_real)
                if dataset.input_view == "compensated":
                    from src.common.numerics import DenseBackend
                    from src.evaluation.inputs import prepare_batch
                    batch = prepare_batch(dataset, si, frames, d, DenseBackend())
                    channels[:] = batch.channel
                    observations[:] = batch.raw_observations
                    variances[:] = batch.variances
                for index, fi in enumerate(frames if dataset.input_view == "raw" else ()):
                    if bases is None:
                        raw_input = dataset.raw_model_input(si, fi)
                        channels[index] = raw_input["csi"][d]
                        observations[index] = raw_input["observations"][d+1]
                        variances[index] = raw_input["noise"]["rf_variance"]
                        del raw_input
                    else:
                        # Receiver-only fields: no reference, labels, alpha or SNR diagnostics.
                        h = dataset._h
                        channels[index] = h["samples/raw_coefficients"][si, fi]
                        variance = float(h["samples/rf_noise_variance"][si, fi])
                        observations[index] = (h["samples/clean_observations"][fi, d+1]
                            + np.sqrt(variance) * h["samples/standard_noise"][fi, d+1])
                        variances[index] = variance
                channel_tensor = torch.as_tensor(channels, dtype=complex_dtype, device=device)
                initial_channel = channel_tensor if bases is None else DPSSChannel(channel_tensor, bases[d])
                inputs = dict(observations=torch.as_tensor(observations, dtype=complex_dtype, device=device),
                    initial_channel=initial_channel,
                    known=known[d].expand(len(frames), -1),
                    data_mask=data_mask, pilot_mask=pilot_mask, null_mask=null_mask,
                    carrier_positions=positions,
                    rf_variance=torch.as_tensor(variances, dtype=real_dtype, device=device),
                    unit_covariance=covariance)
                del channels, observations, variances
                yield inputs, [(si, fi, d) for fi in frames]
                del inputs, initial_channel, channel_tensor


def supervision(dataset, keys, *, device="cpu", include_reference=True, precision="complex64"):
    """Read labels/reference only after model input preparation or inference."""
    complex_dtype, _ = precision_dtypes(precision)
    numpy_complex = np.complex64 if precision == "complex64" else np.complex128
    references, symbols, classes, records = [], [], [], []
    for si, fi, d in keys:
        dataset._check_indices(si, fi)
        if d not in (0, 1):
            raise IndexError(d)
        labels = dataset.labels(fi)
        if include_reference:
            references.append(dataset.reference_csi(fi)[d])
        symbols.append(labels["symbols"][d])
        classes.append(labels["classes"][d])
        wi = int(dataset._h["samples/window_index"][fi])
        records.append(int(dataset._h["windows/record"][wi]))
    result = dict(symbols=torch.as_tensor(np.asarray(symbols, dtype=numpy_complex), dtype=complex_dtype, device=device),
        classes=torch.as_tensor(np.stack(classes), dtype=torch.long, device=device),
        records=torch.as_tensor(records, dtype=torch.long, device=device),
        snr_indices=torch.as_tensor([key[0] for key in keys], dtype=torch.long, device=device),
        frames=torch.as_tensor([key[1] for key in keys], dtype=torch.long, device=device),
        blocks=torch.as_tensor([key[2] for key in keys], dtype=torch.long, device=device))
    if include_reference:
        result["reference_channel"] = torch.as_tensor(np.asarray(references, dtype=numpy_complex), dtype=complex_dtype, device=device)
    return result
