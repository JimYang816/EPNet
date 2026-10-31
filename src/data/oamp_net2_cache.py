"""Content-bound, resumable train/validation spectral caches on disk.

All data carriers are retained. Eigenvectors are complex128 and can be large;
only mini-batches are transferred to the device. Test inputs are never cached
here, and labels are attached by the separate supervision module.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import shutil

import numpy as np

from src.algorithms.likelihood import make_whitener
from src.algorithms.oamp_net2.model import FEATURES
from src.common.experiment import file_sha256, write_json
from src.data.oamp_net2 import PREPROCESSING, PREPROCESS_CODE, spectral_features
from src.evaluation.fingerprints import digest, source_fingerprint, numerical_environment
from src.evaluation.inputs import prepare_batch
from src.evaluation.result_cache import FileLock
from src.training.oamp_net2_supervision import symbols

LOG = logging.getLogger(__name__)
CACHE_CODE = PREPROCESS_CODE+("src.data.oamp_net2_cache", "src.training.oamp_net2_supervision")


class SpectralCache:
    def __init__(self, path, manifest):
        self.path, self.manifest = Path(path), manifest
        self.arrays = {name: np.load(self.path/(name+".npy"), allow_pickle=False, mmap_mode="r")
                       for name in (*FEATURES, "symbols")}
        self.count = len(self.arrays["eigenvalues"])

    def batch(self, indices, device):
        import torch
        return {name: torch.as_tensor(array[indices], device=device) for name, array in self.arrays.items()}

    def close(self):
        self.arrays.clear()


def build_features(dataset, root, backend, data_identity, *, frame_batch_size=8, frame_limit=None):
    if dataset.input_view != "compensated" or dataset.split not in ("train", "validation"):
        raise ValueError("Only compensated train/validation OAMP-Net2 features may be cached")
    if (not isinstance(frame_batch_size, int) or isinstance(frame_batch_size, bool) or frame_batch_size < 1
            or (frame_limit is not None and (not isinstance(frame_limit, int)
                or isinstance(frame_limit, bool) or frame_limit < 1))):
        raise ValueError("Invalid OAMP-Net2 feature batching controls")
    count = min(dataset.frame_count, frame_limit) if frame_limit is not None else dataset.frame_count
    n = len(dataset._indices)
    total = len(dataset.snr_db)*2*count
    shapes = dict(eigenvalues=(total, n), eigenvectors=(total, n, n), projected_matched=(total, n),
        observation_energy=(total,), receive_count=(total,), symbols=(total, n))
    dtypes = {name: np.complex128 if name in ("eigenvectors", "projected_matched", "symbols") else np.float64
              for name in shapes}
    identity = dict(schema="oamp-net2-features-v1", split=dataset.split, input_view=dataset.input_view,
        data={name: data_identity.files[name] for name in (dataset.split+".h5", "model.h5")},
        preprocessing=PREPROCESSING, code=source_fingerprint(CACHE_CODE), frame_limit=frame_limit,
        environment=numerical_environment(backend), count=total, shapes=shapes)
    # Canonicalize tuples once, so JSON reloads compare identically.
    identity = json.loads(json.dumps(identity))
    path = Path(root)/(dataset.split+"_"+digest(identity))
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_files = {name+".npy" for name in shapes}
    with FileLock(path.with_suffix(".lock")):
        manifest = None
        try:
            candidate = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
            if (candidate.get("identity") == identity and candidate.get("complete")
                    and set(candidate["sha256"]) == expected_files
                    and all(file_sha256(path/name) == value for name, value in candidate["sha256"].items())):
                manifest = candidate
        except (OSError, ValueError, KeyError):
            pass
        if manifest is None:
            path.mkdir(exist_ok=True)
            (path/"blocks").mkdir(exist_ok=True)
            allocation = sum(int(np.prod(shape))*np.dtype(dtypes[name]).itemsize for name, shape in shapes.items()
                             if not (path/(name+".npy")).exists())
            if shutil.disk_usage(path).free < allocation+1024**3:
                raise OSError("Insufficient disk for full OAMP-Net2 spectral features (plus 1 GiB reserve)")
            arrays = {}
            try:
                for name, shape in shapes.items():
                    filename = path/(name+".npy")
                    try:
                        array = np.load(filename, mmap_mode="r+", allow_pickle=False)
                        if array.shape != shape or array.dtype != np.dtype(dtypes[name]):
                            raise ValueError("Feature shape/dtype mismatch")
                    except (OSError, ValueError):
                        array = np.lib.format.open_memmap(filename, mode="w+", shape=shape, dtype=dtypes[name])
                    arrays[name] = array
                def checksum(start, stop):
                    hasher = hashlib.sha256()
                    for name in sorted(arrays):
                        hasher.update(name.encode())
                        hasher.update(np.asarray(arrays[name][start:stop]).tobytes())
                    return hasher.hexdigest()
                whitener = make_whitener(dataset._shared["unit_rf_noise_covariance"][:], backend)
                offset = 0
                for si, snr in enumerate(dataset.snr_db):
                    for d in range(2):
                        for start in range(0, count, frame_batch_size):
                            frames = list(range(start, min(start+frame_batch_size, count)))
                            stop = offset+len(frames)
                            marker = path/"blocks"/f"{si}_{d}_{start}_{len(frames)}.json"
                            try:
                                saved = json.loads(marker.read_text(encoding="utf-8"))
                            except (OSError, ValueError):
                                saved = {}
                            if saved.get("identity") != digest(identity) or saved.get("sha256") != checksum(offset, stop):
                                features = spectral_features(prepare_batch(dataset, si, frames, d, backend), whitener, backend)
                                features["symbols"] = symbols(dataset, frames, d)
                                for name, value in features.items():
                                    arrays[name][offset:stop] = value
                                    arrays[name].flush()
                                write_json(marker, dict(identity=digest(identity), sha256=checksum(offset, stop)))
                            offset = stop
                    LOG.info("%s %s OAMP-Net2 features SNR=%g: %d/%d blocks", dataset.channel, dataset.split, snr, offset, total)
                data_identity.assert_unchanged()
                manifest = dict(identity=identity, complete=True, count=total,
                    sha256={name+".npy": file_sha256(path/(name+".npy")) for name in shapes})
                write_json(path/"manifest.json", manifest)
            finally:
                arrays.clear()
        data_identity.assert_unchanged()
        return SpectralCache(path, manifest)
