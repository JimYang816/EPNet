"""Official replayfilter.m interpolation, using MAT metadata as the time grid.

The measured Doppler stays in h. V0 is recorded, never compensated. No gain
normalization or conjugation is applied to the measured impulse responses.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from functools import lru_cache
import hashlib
import zipfile
import numpy as np
from scipy.io import loadmat
from scipy import sparse


@dataclass
class Sounding:
    h: np.ndarray
    fs_time: float
    fs_delay: int
    carrier_hz: int
    v0: float
    source: str
    sha256: str = "synthetic"

    @property
    def duration(self):
        return (len(self.h)-1)/self.fs_time

    def interpolate(self, times):
        p = np.asarray(times)*self.fs_time
        if np.min(p) < 0 or np.max(p) >= len(self.h)-1:
            raise ValueError(f"Window crosses record boundary: {self.source}")
        i = np.floor(p).astype(int)
        f = p-i
        return (1-f[:, None])*self.h[i] + f[:, None]*self.h[i+1]

    def operator(self, length, start_seconds):
        h = self.interpolate(start_seconds+np.arange(length)/self.fs_delay)
        rows = np.broadcast_to(np.arange(length)[:, None], h.shape)
        cols = rows-np.arange(self.h.shape[1])
        valid = cols >= 0
        return sparse.csr_matrix((h[valid], (rows[valid], cols[valid])), shape=(length, length))

    def metadata(self):
        return dict(source=self.source, sha256=self.sha256, shape=list(self.h.shape),
                    fs_time_hz=self.fs_time, fs_delay_hz=self.fs_delay, carrier_hz=self.carrier_hz,
                    v0=self.v0, time_axis_seconds=[0, self.duration],
                    delay_axis_seconds=[0, (self.h.shape[1]-1)/self.fs_delay])


class WatermarkArchive:
    def __init__(self, path, channel="NCS1"):
        if channel not in ("NCS1", "NOF1"):
            raise ValueError("Supported WATERMARK channels are NCS1 and NOF1")
        self.path = Path(path)
        self.channel = channel

    @lru_cache(maxsize=2)
    def load(self, number):
        filename = f"{self.channel}_{number:03d}.mat"
        with zipfile.ZipFile(self.path) as z:
            names = [n for n in z.namelist() if n.endswith(f"/{self.channel}/mat/"+filename)]
            if len(names) != 1:
                raise FileNotFoundError(filename)
            content = z.read(names[0])
        m = loadmat(BytesIO(content))
        scalar = lambda k: float(np.asarray(m[k]).item())
        h = np.asarray(m["h"], np.complex128)
        fs_t, fs_tau, fc = scalar("fs_t"), scalar("fs_tau"), scalar("fc")
        if h.ndim != 2 or not np.all(np.isfinite(h)):
            raise ValueError("Invalid measured TVIR")
        if (fs_tau, fc) != (16000, 14000) or not np.isclose(fs_t, fs_tau/h.shape[1], rtol=1e-10):
            raise ValueError(f"Unexpected {self.channel} metadata; inspect the actual axes before replay")
        return Sounding(h, fs_t, int(fs_tau), int(fc), scalar("V0"), names[0], hashlib.sha256(content).hexdigest())

    def audit(self, records):
        file_hash = hashlib.sha256()
        with self.path.open("rb") as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b""):
                file_hash.update(chunk)
        return dict(channel=self.channel, archive_sha256=file_hash.hexdigest(),
                    records={str(r): self.load(r).metadata() for r in records})


def direct_convolution(x, sounding, start_seconds):
    """Literal independent reference for the official causal convolution loop."""
    x = np.asarray(x, np.complex128)
    h = sounding.interpolate(start_seconds + np.arange(len(x))/sounding.fs_delay)
    padded = np.pad(x, (h.shape[1]-1, 0))
    return np.asarray([padded[k:k+h.shape[1]] @ h[k, ::-1] for k in range(len(x))])
