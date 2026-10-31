"""Label-free full-matrix spectral sufficient statistics for OAMP-Net2."""
from __future__ import annotations

import numpy as np

from src.algorithms.likelihood import batch_likelihood

PREPROCESSING = dict(version=1, input_view="compensated", support="all_receive_rows_all_data_columns",
    pilot_cancellation="complete_estimated_channel", noise="full_thermal_covariance",
    whitening="unit_covariance_and_rf_variance", decomposition="full_complex_hermitian_eigh",
    precision="complex128/float64", csi="fixed_no_error_oracle")
PREPROCESS_CODE = ("src.data.oamp_net2", "src.algorithms.likelihood:batch_likelihood",
    "src.algorithms.likelihood:likelihood_batch", "src.algorithms.likelihood:make_whitener",
    "src.evaluation.inputs", "src.data.compact:CompactDataset", "src.data.contract",
    "src.common.numerics", "src.algorithms.interface", "data_generation.src.receiver.dpss:DPSSModel")


def spectral_features(batch, whitener, backend):
    if np.any(~np.isfinite(batch.variances)) or np.any(batch.variances <= 0):
        raise ValueError("OAMP-Net2 requires finite positive RF noise variances")
    gram, matched = batch_likelihood(batch, whitener, backend)
    t = backend.torch
    adjoint = gram.mH if t else gram.conj().swapaxes(-1, -2)
    symmetric = (gram+adjoint)*.5
    if t:
        values, vectors = t.linalg.eigh(symmetric)
        projected = (vectors.mH@matched[..., None])[..., 0]
        y = batch.cancelled_observations
        y = y if isinstance(y, t.Tensor) else backend.array(y)
        white_y = (whitener@y[..., None])[..., 0]
        energy = (white_y.abs().square()).sum(dim=-1)/backend.array(batch.variances)
    else:
        pairs = [backend.eigh(item) for item in symmetric]
        values, vectors = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        projected = (vectors.conj().swapaxes(-1, -2)@matched[..., None])[..., 0]
        white_y = (whitener@np.asarray(batch.cancelled_observations)[..., None])[..., 0]
        energy = np.sum(abs(white_y)**2, axis=-1)/batch.variances
    values, vectors = backend.numpy(values), backend.numpy(vectors)
    if np.any(values < -1e-10*np.maximum(1., np.max(abs(values), axis=-1, keepdims=True))):
        raise ValueError("OAMP-Net2 Gram matrix is not positive semidefinite")
    features = dict(eigenvalues=np.maximum(values, 0.), eigenvectors=vectors,
        projected_matched=backend.numpy(projected), observation_energy=backend.numpy(energy),
        receive_count=np.full(len(batch.variances), batch.channel.shape[-2], dtype=np.float64))
    if any(not np.isfinite(value).all() for value in features.values()):
        raise FloatingPointError("Nonfinite OAMP-Net2 features")
    return features
