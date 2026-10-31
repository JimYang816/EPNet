"""Lazy unified learning-data inputs and a separate explicit supervision/audit view."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from data_generation.src.dataset.compact_protocol import SCHEMA, MODEL_SCHEMA
from data_generation.src.dataset.compact_math import reconstruct
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.transmitter.ofdm import qpsk
from src.data.uncertainty import CovarianceProxy
from src.data.contract import DEFAULT_INPUT_VIEW, validate_compensated_contract


def data_system(inputs):
    """One pilot-cancellation/data-column contract for every future equalizer."""
    channel, known = inputs["csi"], inputs["known"][[1, 2]]
    y = inputs["observations"][[1, 2]]
    cancelled = y-np.einsum("dij,dj->di", channel, known, optimize=True)
    return cancelled, channel[..., inputs["data_indices"]]


class CompactDataset:
    """Each item is (model_inputs, labels); reference information requires audit().

    Open one reader per worker. Covariance interfaces remain valid while the
    reader is open; gain matrices are streamed from the shared model file.
    """

    def __init__(self, path, *, input_view=DEFAULT_INPUT_VIEW):
        if input_view not in ("raw", "compensated"):
            raise ValueError("Unknown CSI input view")
        self.input_view = input_view
        self.path = Path(path)
        self._h = h5py.File(self.path, "r")
        self._shared = None
        try:
            h = self._h
            if h.attrs.get("schema") != SCHEMA:
                raise ValueError("Compact dataset protocol mismatch; historical v1 data cannot be mixed")
            if (not h.attrs.get("complete") or not h.attrs.get("compensated")
                    or not np.all(h["windows/complete"][:]) or not np.all(h["samples/complete"][:])):
                raise ValueError("Compact dataset is incomplete")
            if h.attrs.get("model_file") != "model.h5":
                raise ValueError("Invalid shared model filename")
            self._shared = h5py.File(self.path.parent/"model.h5", "r")
            m = self._shared
            if (m.attrs.get("schema") != MODEL_SCHEMA or not m.attrs.get("complete")
                    or not m.attrs.get("training_calibration_complete")
                    or m.attrs.get("identity") != h.attrs.get("model_identity")):
                raise ValueError("Shared model identity/protocol/incomplete mismatch")
            self._config = json.loads(m.attrs["config_json"])
            c = self._config
            self._model = DPSSModel(m["basis"][:], c["dpss_half_bandwidth_hz"], c["dpss_tail_ratio"], c["delay_length"], c)
            self._blocks = [(int(g.attrs["start"]), int(g.attrs["stop"]), g["temporal"][:], g["delay"][:])
                            for g in m["prior/blocks"].values()]
            self.snr_db = h["snr_db"][:]
            self.channel, self.split = h.attrs["channel"], h.attrs["split"]
            if self.input_view == DEFAULT_INPUT_VIEW:
                validate_compensated_contract(self)
            self._known = m["known"][:]
            self._masks = {k: v[:] for k, v in m["masks"].items()}
            self._indices = np.flatnonzero(self._masks["data"])
            nf, ns, n = self.frame_count, len(self.snr_db), c["nfft"]
            nw = len(h["windows/record"])
            # Raw input validation never accesses supervision/compensation fields.
            if (h["samples/raw_coefficients"].shape != (ns, nf, self._model.count, self._model.length)
                    or h["samples/clean_observations"].shape != (nf, 4, n)
                    or not np.array_equal(self.snr_db, m["snr_db"][:])
                    or np.any(h["samples/window_index"][:] < 0) or np.any(h["samples/window_index"][:] >= nw)):
                raise ValueError("Invalid compact frame dimensions or window indices")
        except BaseException:
            self.close()
            raise

    @property
    def frame_count(self):
        return len(self._h["samples/window_index"])

    def __len__(self):
        return self.frame_count*len(self.snr_db)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        si, fi = divmod(index, self.frame_count)
        return self.model_input(si, fi), self.labels(fi)

    def _check_indices(self, si, fi):
        if not 0 <= si < len(self.snr_db) or not 0 <= fi < self.frame_count:
            raise IndexError((si, fi))

    def _channels(self, si, fi):
        h = self._h
        wi = int(h["samples/window_index"][fi])
        reference = h["windows/reference_csi"][wi].astype(complex)
        coefficients = h["samples/raw_coefficients"][si, fi].astype(complex)
        raw = reconstruct(self._model, coefficients)
        alpha = float(h["compensation/alpha"][si])
        return reference, raw, reference+alpha*(raw-reference), coefficients, alpha

    def _proxy(self, si, fi, scale):
        m = self._shared
        return CovarianceProxy(self._blocks, m["estimator/gain"], m["estimator/values"][:],
                               m["estimator"].attrs["ridge"], self._h["samples/effective_rf_variance"][si, fi],
                               (self._model.count, self._model.length), scale)

    def model_input(self, snr_index, frame_index):
        if self.input_view == "raw":
            return self.raw_model_input(snr_index, frame_index)
        return self.compensated_model_input(snr_index, frame_index)

    def raw_model_input(self, snr_index, frame_index):
        """Receiver-only input: no reference CSI, labels, alpha or audit reads."""
        si, fi = snr_index, frame_index
        self._check_indices(si, fi)
        h, m = self._h, self._shared
        coefficients = h["samples/raw_coefficients"][si, fi].astype(complex)
        raw = reconstruct(self._model, coefficients)
        variance = float(h["samples/rf_noise_variance"][si, fi])
        observations = h["samples/clean_observations"][fi]+np.sqrt(variance)*h["samples/standard_noise"][fi]
        return dict(input_view="raw", observations=observations, csi=raw,
                    noise=dict(rf_variance=variance, unit_rx_covariance=m["unit_rf_noise_covariance"][:]),
                    known=self._known.copy(), masks={k: v.copy() for k, v in self._masks.items()},
                    data_indices=self._indices.copy(), pilot_indices=np.flatnonzero(self._masks["pilot"]),
                    carrier_positions=np.arange(self._config["nfft"]))

    def compensated_model_input(self, snr_index, frame_index):
        """Accepted default CSI with the user's fixed per-SNR NMSE targets."""
        si, fi = snr_index, frame_index
        self._check_indices(si, fi)
        h, m = self._h, self._shared
        _, _, compensated, _, alpha = self._channels(si, fi)
        variance = float(h["samples/rf_noise_variance"][si, fi])
        observations = h["samples/clean_observations"][fi]+np.sqrt(variance)*h["samples/standard_noise"][fi]
        gamma = float(m["training_calibration/gamma"][si])
        # Only calibrated interfaces/powers enter model inputs; no audit fields.
        uncertainty = dict(coefficient_covariance=self._proxy(si, fi, alpha**2*gamma),
                           dpss_basis=m["basis"][:], delay_length=self._model.length,
                           sampled_dpss_basis=self._model.sampled_basis.reshape(4, self._config["nfft"], self._model.count)[[1, 2]],
                           coefficient_shape=(self._model.count, self._model.length),
                           coefficient_order="q_major_delay_minor",
                           fft_convention="fftshift, orthonormal complex FFT",
                           csi_residual_power=alpha**2*m["training_calibration/projection_residual_power"][:],
                           model_residual_power=m["training_calibration/model_residual_power"][:],
                           semantics="receiver_model_covariance_proxy")
        return dict(input_view="compensated", observations=observations, csi=compensated,
                    noise=dict(rf_variance=variance, unit_rx_covariance=m["unit_rf_noise_covariance"][:],
                               actual_snr_db=h["samples/actual_snr_db"][si, fi]),
                    known=self._known.copy(), masks={k: v.copy() for k, v in self._masks.items()},
                    data_indices=self._indices.copy(), pilot_indices=np.flatnonzero(self._masks["pilot"]),
                    uncertainty=uncertainty)

    def labels(self, frame_index):
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(frame_index)
        bits = self._h["samples/bits"][frame_index]
        return dict(bits=bits, symbols=qpsk(bits), classes=(2*bits[..., 0]+bits[..., 1]).astype(np.uint8),
                    constellation=self._shared["constellation/symbols"][:],
                    constellation_bits=self._shared["constellation/bits"][:])

    def reference_csi(self, frame_index):
        """Separate supervision/scoring interface; never called by input readers."""
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(frame_index)
        wi = int(self._h["samples/window_index"][frame_index])
        return self._h["windows/reference_csi"][wi].astype(complex)

    def audit(self, snr_index, frame_index):
        si, fi = snr_index, frame_index
        self._check_indices(si, fi)
        h = self._h
        wi = int(h["samples/window_index"][fi])
        reference, raw, comp, coefficients, alpha = self._channels(si, fi)
        error = comp-reference
        return dict(reference_csi=reference, raw_csi=raw, compensated_csi=comp, true_error=error,
                    raw_coefficients=coefficients, compensation_alpha=alpha,
                    true_nmse_linear=np.sum(abs(error)**2, axis=(1, 2))/np.sum(abs(reference)**2, axis=(1, 2)),
                    model_residual=h["samples/model_residual"][fi],
                    reference_projection=h["windows/reference_projection"][wi],
                    raw_covariance_proxy=self._proxy(si, fi, 1.0),
                    identity=dict(channel=self.channel, split=self.split, record=int(h["windows/record"][wi]),
                                  start_seconds=float(h["windows/start_seconds"][wi]),
                                  payload_repeat=int(h["samples/payload_repeat"][fi]),
                                  noise_repeat=int(h["samples/noise_repeat"][fi]),
                                  payload_seed=h["samples/payload_seed"][fi], noise_seed=h["samples/noise_seed"][fi]))

    def close(self):
        if getattr(self, "_shared", None) is not None:
            self._shared.close()
        self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
