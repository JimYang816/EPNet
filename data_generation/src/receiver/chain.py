"""96-kHz real-passband replay and independently assembled linear operators.

Every resampler uses the MATLAB-style half-length-10 Kaiser(beta=5) prototype
and compensates its deterministic group delay. The FFT starts 32 baseband
samples early. A and B describe the entire real chain, y = A x + B x*.
"""
from __future__ import annotations

from math import gcd, ceil
import numpy as np
from scipy import sparse, signal, linalg

from data_generation.src.transmitter.ofdm import window_indices, receive_symbols
from data_generation.src.pdd import SYMBOL_COUNT, DATA_SYMBOL_INDICES


def fir(up, down, c):
    div = gcd(up, down)
    up, down = up//div, down//div
    half = c["resample_half_length"]*max(up, down)
    return signal.firwin(2*half+1, 1/max(up, down), window=("kaiser", c["resample_kaiser_beta"]))


def resample(x, up, down, c):
    return signal.resample_poly(x, up, down, axis=0, window=fir(up, down, c))


def resample_operator(n, up, down, c):
    div = gcd(up, down)
    up, down = up//div, down//div
    h = fir(up, down, c)*up
    half = (len(h)-1)//2
    row = np.arange(ceil(n*up/down))[:, None]
    col = np.ceil((row*down-half)/up).astype(int) + np.arange(2*half//up+2)
    index = half+row*down-col*up
    valid = (col >= 0) & (col < n) & (index >= 0) & (index < len(h))
    rows = np.broadcast_to(row, col.shape)
    return sparse.csr_matrix((h[index[valid]], (rows[valid], col[valid])), shape=(len(row), n))


def frequency_matrix(a, conjugate_input=False):
    a = np.fft.fft(a, axis=0, norm="ortho")
    a = (np.fft.fft if conjugate_input else np.fft.ifft)(a, axis=1, norm="ortho")
    return np.fft.fftshift(a, axes=(0, 1))


def taps_to_matrix(taps):
    n, length = taps.shape
    rows = np.arange(n)[:, None]
    a = np.zeros((n, n), np.complex128)
    a[rows, (rows-np.arange(length)) % n] = taps
    return frequency_matrix(a)


def full_frame_kernels(operator, c, length):
    """Sample the physical operator on all CP/useful rows, with zero extension.

    Delay zero uses the same advance convention as the FFT-window kernels.
    At L=384 early-frame input columns can be negative; NumPy wrapping would
    incorrectly borrow samples from the end of the operator.
    """
    if length < 1:
        raise ValueError("Delay support must be positive")
    total = SYMBOL_COUNT*(c["nfft"]+c["cp"])
    rows = np.broadcast_to(c["guard_samples"]+np.arange(total)[:, None], (total, length))
    cols = rows+c["fft_advance"]-np.arange(length)
    valid = (rows >= 0) & (rows < operator.shape[0]) & (cols >= 0) & (cols < operator.shape[1])
    result = np.zeros((total, length), np.complex128)
    selected = operator[rows[valid], cols[valid]]
    result[valid] = np.asarray(selected.toarray() if hasattr(selected, "toarray") else selected).ravel()
    return result


def relative_energy(error, reference):
    return float(np.sum(abs(error)**2)/max(np.sum(abs(reference)**2), 1e-300))


class ReceiverChain:
    def __init__(self, c):
        self.c = c
        self.n = c["nfft"]
        self.length = SYMBOL_COUNT*(self.n+c["cp"]) + 2*c["guard_samples"]
        u = resample_operator(self.length, 16, 1, c)
        rin = resample_operator(u.shape[0], 1, 6, c)
        # ceil(N/6)*6 may exceed N for a four-symbol frame. Match the
        # finite transmitted RF interval after returning from channel rate.
        rout = resample_operator(rin.shape[0], 6, 1, c)[:u.shape[0]]
        rx = resample_operator(rout.shape[0], 1, 16, c)
        t = np.arange(u.shape[0])/c["fs_passband"]
        phase = np.exp(-4j*np.pi*c["carrier_hz"]*t)
        self.u = (rin @ u).tocsr()
        self.ui = (rin @ sparse.diags(phase) @ u).tocsr()
        self.r = (rx @ rout).tocsr()
        self.ri = (rx @ sparse.diags(phase) @ rout).tocsr()
        self.phase_carrier = np.exp(2j*np.pi*c["carrier_hz"]*t)
        h = fir(1, 16, c)
        ac = signal.correlate(h, h, mode="full")
        lag = np.arange(self.n)*16
        col = np.zeros(self.n)
        keep = lag < len(h)
        col[keep] = 4*ac[len(h)-1+lag[keep]]
        self.noise_covariance = frequency_matrix(linalg.toeplitz(col))
        self.noise_gain = float(col[0])

    def transmit(self, baseband):
        return np.real(resample(baseband, 16, 1, self.c)*self.phase_carrier)

    def receive(self, passband):
        return 2*resample(passband*self.phase_carrier.conj(), 1, 16, self.c)

    def replay(self, tx_passband, sounding, start_seconds):
        if sounding.carrier_hz != self.c["carrier_hz"]:
            raise ValueError("Incompatible carrier frequencies")
        x = resample(tx_passband*self.phase_carrier.conj(), 1, 6, self.c)
        z = sounding.operator(len(x), start_seconds) @ x
        return np.real(resample(z, 6, 1, self.c)[:len(tx_passband)]*self.phase_carrier)

    def operators(self, sounding, start_seconds):
        h = sounding.operator(self.u.shape[0], start_seconds)
        rh, rih = self.r @ h, self.ri @ h.conjugate()
        a = (0.5*(rh @ self.u + rih @ self.ui.conjugate())).tocsr()
        b = (0.5*(rh @ self.ui + rih @ self.u)).tocsr()
        a.eliminate_zeros()
        b.eliminate_zeros()
        return a, b

    def symbol_operator(self, op, symbol):
        part = op[window_indices(self.c, symbol)].tocoo()
        begin = self.c["guard_samples"]+symbol*(self.n+self.c["cp"])
        offset = part.col-begin
        inside = (offset >= 0) & (offset < self.n+self.c["cp"])
        a = np.zeros((self.n, self.n), np.complex128)
        np.add.at(a, (part.row[inside], (offset[inside]-self.c["cp"]) % self.n), part.data[inside])
        leakage = float(np.sum(abs(part.data[~inside])**2)/max(np.sum(abs(part.data)**2), 1e-300))
        return a, leakage

    def reference(self, a, symbol):
        circular, leakage = self.symbol_operator(a, symbol)
        rows = np.arange(self.n)[:, None]
        taps = circular[rows, (rows-np.arange(self.n)) % self.n]
        return frequency_matrix(circular), taps, leakage

    def frame(self, sounding, start, frame, *, operators=None):
        # start identifies the beginning of the first CP. Extra quiet samples
        # on both sides keep all finite FIR supports inside the measured record.
        origin = start-self.c["guard_samples"]/self.c["fs_baseband"]
        clean = self.replay(self.transmit(frame.baseband), sounding, origin)
        ybb = self.receive(clean)
        a, b = self.operators(sounding, origin) if operators is None else operators
        linear_error = relative_energy(ybb-a@frame.baseband-b@frame.baseband.conj(), ybb)
        y = receive_symbols(ybb, self.c)
        kernels, matrices, leaks, images, predictions = [], [], [], [], []
        for s in range(SYMBOL_COUNT):
            h, g, leak = self.reference(a, s)
            ib, _ = self.symbol_operator(b, s)
            ca, _ = self.symbol_operator(a, s)
            kernels.append(g)
            matrices.append(h)
            leaks.append(leak)
            images.append(relative_energy(ib, ca))
            predictions.append(relative_energy(y[s]-h@frame.x[s], y[s]))
        diagnostics = dict(waveform_operator_error=linear_error, prediction_errors=predictions,
                           image_ratios=images, intersymbol_leakage=leaks)
        limit = 10**(self.c["link_error_db"]/10)
        diagnose = self.c.get("model_mismatch_policy", "reject") == "diagnose"
        model_passed = max(predictions+leaks) <= limit
        diagnostics.update(model_mismatch_policy="diagnose" if diagnose else "reject",
                           circular_model_passed=model_passed,
                           model_mismatch_retained=diagnose and not model_passed)
        if max([linear_error]+images) > limit or (not model_passed and not diagnose):
            raise RuntimeError(f"Physical link acceptance failed: {diagnostics}")
        reference = np.stack([matrices[s] for s in DATA_SYMBOL_INDICES])
        return clean, y, np.stack(kernels), reference, diagnostics

    def noise(self, clean, record, start, split=2, *, noise_repeat=None):
        seed = [self.c["seed"], split, record, round(start*1000), 1]
        if noise_repeat is not None:
            seed.append(int(noise_repeat))
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        standard = rng.standard_normal(len(clean))
        standard_y = receive_symbols(self.receive(standard), self.c)
        symbols = DATA_SYMBOL_INDICES
        power = np.mean([np.mean(abs(self.receive(clean)[window_indices(self.c, s)])**2) for s in symbols])
        variances = power/self.noise_gain/10**(np.asarray(self.c["snr_db"])/10)
        return standard_y, variances, float(power)

    def validate_noise(self, repeats=64):
        """Independent real-RF repeats check gain, correlations and propriety."""
        rng = np.random.default_rng(np.random.SeedSequence([self.c["seed"], 99]))
        z = np.stack([self.receive(rng.standard_normal(len(self.phase_carrier)))[128:-128] for _ in range(repeats)])
        gain = float(np.mean(abs(z)**2))
        h = fir(1, 16, self.c)
        ac = signal.correlate(h, h, mode="full")
        checks = []
        for lag in (1, 2, 5, 10):
            predicted = 4*ac[len(h)-1+16*lag]
            measured = np.mean(z[:, lag:]*z[:, :-lag].conj())
            checks.append(dict(lag=lag, predicted=float(predicted), measured_real=float(measured.real),
                               measured_imag=float(measured.imag), normalized_error=float(abs(measured-predicted)/self.noise_gain)))
        error = abs(gain/self.noise_gain-1)
        pseudo = float(abs(np.mean(z*z))/gain)
        result = dict(repeats=repeats, seed_suffix=99, analytical_gain=self.noise_gain, measured_gain=gain,
                      gain_relative_error=error, normalized_pseudocovariance=pseudo, covariance_lags=checks,
                      passed=bool(error < .02 and pseudo < .015 and max(x["normalized_error"] for x in checks) < .015))
        if not result["passed"]:
            raise RuntimeError(f"Independent receiver noise validation failed: {result}")
        return result
