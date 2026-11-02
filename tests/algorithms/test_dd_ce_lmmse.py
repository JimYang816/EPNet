"""Compare DD-CE against explicit time-varying channel matrices and a receive-space solve."""
from types import SimpleNamespace

import numpy as np
import pytest

from data_generation.src.receiver.chain import taps_to_matrix
from src.algorithms.dd_ce_lmmse.detector import build_modes, equalize_batch
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.numerics import DenseBackend


@pytest.mark.parametrize("device", ["numpy", "torch_cpu", "cuda"])
def test_time_varying_dd_update_and_redetection(device):
    backend = DenseBackend()
    if device != "numpy":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA hardware unavailable")
        backend = SimpleNamespace(torch=torch,
            array=lambda x: torch.as_tensor(x, device="cuda" if device == "cuda" else "cpu"),
            numpy=lambda x: x.detach().cpu().numpy())
    rng = np.random.default_rng(31)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    n = 9
    basis = random((n, 3))
    blocks = []
    physical_covariance = np.zeros((n*n, n*n), complex)
    for start, end in [(0, 2), (3, 6)]:
        t, d = random((3, 3)), random((end-start, end-start))
        temporal, delay = t@t.conj().T, d@d.conj().T
        blocks.append((start, end, temporal, delay))
        padded = np.zeros((n, n), complex)
        padded[start:end, start:end] = delay
        physical_covariance += np.kron(basis@temporal@basis.conj().T, padded)
    modes = build_modes(basis, blocks, 4, backend)
    eig, vec = np.linalg.eigh(physical_covariance)
    explicit = [taps_to_matrix((np.sqrt(eig[i])*vec[:, i]).reshape(n, n)) for i in range(-4, 0)]
    h = np.stack([np.eye(n)+.08*random((n, n)) for _ in range(2)])
    data = np.array([0, 2, 4, 6])
    known = np.zeros(n, complex)
    known[[1, 5]] = [1j, -1]
    y = random((2, n))
    a = random((n, n))
    noise = a@a.conj().T/n+np.eye(n)
    variances, prior_scale = np.array([.1, .3]), .03
    b = backend.array
    batch = EqualizerBatch(b(h), b(y), b(y-h@known), variances, data, noise)
    result = equalize_batch(batch, backend, whitener=make_whitener(noise, backend),
                            known=known, modes=modes, prior_scale=prior_scale)
    for i, (initial, observation, variance) in enumerate(zip(h, y, variances)):
        def detect(channel):
            columns = channel[:, data]
            return columns.conj().T@np.linalg.solve(columns@columns.conj().T + variance*noise,
                                                    observation-channel@known)
        first = detect(initial)
        x = known.copy()
        x[data] = (np.where(first.real < 0, -1, 1)+1j*np.where(first.imag < 0, -1, 1))/np.sqrt(2)
        design = np.stack([mode@x for mode in explicit], axis=1)
        # Independent Bayesian receive-space update; no whitening or normal-equation code.
        weights = design.conj().T@np.linalg.solve(design@design.conj().T+variance/prior_scale*noise,
                                                  observation-initial@x)
        expected = initial+sum(w*m for w, m in zip(weights, explicit))
        np.testing.assert_allclose(backend.numpy(result.corrected_channel)[i], expected, atol=3e-10)
        np.testing.assert_allclose(result.symbols[i], detect(expected), atol=3e-10)
        assert not np.allclose(detect(expected), first)
    np.testing.assert_array_equal(backend.numpy(batch.channel), h)


def test_scored_nmse_uses_full_matching_reference_for_each_frame_and_d():
    from src.evaluation.dd_ce import corrected_nmse
    rng = np.random.default_rng(39)
    truth = rng.normal(size=(3, 2, 5, 5))+1j*rng.normal(size=(3, 2, 5, 5))
    windows = np.array([2, 0, 2, 1])
    dataset = SimpleNamespace(_h={"samples/window_index": windows, "windows/reference_csi": truth})
    for d in (0, 1):
        estimated = truth[windows, d].copy()
        estimated[:, 3, 1] += np.array([1j, -2j, 0, 3])
        expected = [np.linalg.norm(h-truth[w, d])**2/np.linalg.norm(truth[w, d])**2
                    for h, w in zip(estimated, windows)]
        np.testing.assert_allclose(corrected_nmse(dataset, list(range(4)), d, estimated, DenseBackend()), expected)
