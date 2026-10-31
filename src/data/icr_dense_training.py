"""Selected-view CSI construction with full reference support on the device.

Reproduce the legacy complex128 tap expansion and 2-D FFT, then cast the
complete matrix to the requested model precision. No operator approximation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.data.icr_mmse import precision_dtypes


@dataclass(frozen=True)
class DenseCoefficients:
    coefficients: torch.Tensor
    basis: torch.Tensor
    dtype: torch.dtype
    reference: torch.Tensor | None = None
    alpha: float = 1.

    def dense(self, device, *, basis=None, reference=None, non_blocking=False):
        values = (self.basis.to(device=device, non_blocking=non_blocking)
                  if basis is None else basis)
        coefficients = self.coefficients.to(device=device, non_blocking=non_blocking)
        n, length = values.shape[0], coefficients.shape[-1]
        taps = torch.einsum("nq,bql->bnl", values, coefficients)
        rows = torch.arange(n, device=device)[:, None]
        columns = (rows-torch.arange(length, device=device)) % n
        matrix = torch.zeros(len(coefficients), n, n, dtype=torch.complex128, device=device)
        matrix.scatter_(-1, columns[None].expand(len(coefficients), -1, -1), taps)
        del taps
        matrix = torch.fft.fft(matrix, dim=-2, norm="ortho")
        matrix = torch.fft.ifft(matrix, dim=-1, norm="ortho")
        matrix = torch.fft.fftshift(matrix, dim=(-2, -1))
        if self.reference is not None:
            reference = (self.reference if reference is None else reference).to(
                device=device, dtype=torch.complex128, non_blocking=non_blocking)
            matrix = reference + self.alpha*(matrix-reference)
        return matrix.to(self.dtype)


def dense_input_batches(dataset, *, batch_size, frame_limit=None, precision="complex64",
                        snr_indices=None, frame_indices=None, reference_cache=None):
    if dataset.input_view not in ("raw", "compensated"):
        raise ValueError("Unknown ICR CSI input view")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Invalid ICR batch size")
    dtype, real = precision_dtypes(precision)
    numpy_complex = np.complex64 if precision == "complex64" else np.complex128
    numpy_real = np.float32 if precision == "complex64" else np.float64
    count = dataset.frame_count if frame_limit is None else min(frame_limit, dataset.frame_count)
    frames = list(range(count)) if frame_indices is None else list(frame_indices)
    selected = list(range(len(dataset.snr_db))) if snr_indices is None else list(snr_indices)
    if (not frames or len(set(frames)) != len(frames)
            or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                   or not 0 <= i < dataset.frame_count for i in frames)
            or any(not 0 <= i < len(dataset.snr_db) for i in selected)):
        raise ValueError("Invalid ICR sample indices")
    n, sampled = dataset._config["nfft"], dataset._model.sampled_basis
    bases = [torch.as_tensor(np.asarray(sampled[(d+1)*n:(d+2)*n], dtype=np.complex128)) for d in range(2)]
    masks = {key: torch.as_tensor(value, dtype=torch.bool) for key, value in dataset._masks.items()}
    shared = dict(data_mask=masks["data"], pilot_mask=masks["pilot"],
        null_mask=~(masks["data"] | masks["pilot"]), carrier_positions=torch.arange(n),
        unit_covariance=torch.as_tensor(np.asarray(dataset._shared["unit_rf_noise_covariance"][:],
                                                    dtype=numpy_complex), dtype=dtype))
    known = [torch.as_tensor(np.asarray(dataset._known[d+1], dtype=numpy_complex), dtype=dtype) for d in range(2)]
    h = dataset._h
    for si in selected:
        for block in range(2):
            for start in range(0, len(frames), batch_size):
                indices = frames[start:start+batch_size]
                coefficients, observations, variances = [], [], []
                for frame in indices:
                    coefficients.append(np.asarray(h["samples/raw_coefficients"][si, frame], dtype=np.complex128))
                    variance = float(h["samples/rf_noise_variance"][si, frame])
                    observations.append(h["samples/clean_observations"][frame, block+1]
                        + np.sqrt(variance)*h["samples/standard_noise"][frame, block+1])
                    variances.append(variance)
                reference, alpha = None, 1.
                if dataset.input_view == "compensated":
                    windows = h["samples/window_index"][indices]
                    if reference_cache is None:
                        reference = torch.as_tensor(np.stack([
                            np.asarray(h["windows/reference_csi"][int(window), block], dtype=np.complex128)
                            for window in windows]))
                    else:
                        # Saved complex64 values promote exactly to complex128
                        # on the device. Reuse the supervision cache across SNRs
                        # and epochs, without changing the compensation formula.
                        reference = torch.cat([reference_cache.get_full(dataset,
                            int(window), block, precision) for window in windows])
                    alpha = float(h["compensation/alpha"][si])
                inputs = dict(shared, initial_channel=DenseCoefficients(torch.as_tensor(np.stack(coefficients)),
                    bases[block], dtype, reference, alpha), observations=torch.as_tensor(np.asarray(observations, dtype=numpy_complex)),
                    rf_variance=torch.as_tensor(np.asarray(variances, dtype=numpy_real), dtype=real),
                    known=known[block].expand(len(indices), -1))
                yield inputs, [(si, frame, block) for frame in indices]
