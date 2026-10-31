"""Disposable, content-bound FP32 features; test features are never cached."""
from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
import shutil

import numpy as np

from src.common.experiment import file_sha256, write_json
from src.data.udnet import FEATURES, PREPROCESSING, LocalPreprocessor, WindowPlan
from src.evaluation.fingerprints import digest, source_fingerprint
from src.evaluation.inputs import prepare_batch
from src.evaluation.result_cache import FileLock
from src.training.udnet_supervision import window_labels

LOG = logging.getLogger(__name__)
PREPROCESS_CODE = ("src.data.udnet", "src.evaluation.inputs", "src.data.compact",
                   "src.common.numerics", "data_generation.src.receiver.dpss:DPSSModel")
CACHE_CODE = PREPROCESS_CODE+("src.data.udnet_cache", "src.training.udnet_supervision")


def remove_features(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Feature cleanup must stay below the experiment feature root")
    if path.exists():
        shutil.rmtree(path)


class FeatureCache:
    def __init__(self, path, manifest):
        self.path, self.manifest = Path(path), manifest
        # Keep static statistics on the host/disk; transfer only each batch.
        self.arrays = {name: np.load(self.path/(name+".npy"), allow_pickle=False, mmap_mode="r")
                       for name in (*FEATURES, "labels")}
        self.count = len(self.arrays["gram"])

    def batch(self, indices, device):
        import torch
        return {name: torch.as_tensor(array[indices], device=device) for name, array in self.arrays.items()}


def build_features(dataset, root, backend, data_identity, *, frame_batch_size=8, frame_limit=None):
    if dataset.split not in ("train", "validation"):
        raise ValueError("Only train/validation local features may be cached")
    if frame_batch_size < 1 or (frame_limit is not None and frame_limit < 1):
        raise ValueError("Invalid feature batching controls")
    plan = WindowPlan.build(dataset._config["nfft"], dataset._indices)
    count = min(dataset.frame_count, frame_limit) if frame_limit is not None else dataset.frame_count
    identity = dict(schema="udnet-features-v2", split=dataset.split, input_view=dataset.input_view,
                    data={name: data_identity.files[name] for name in (dataset.split+".h5", "model.h5")},
                    preprocessing=PREPROCESSING, code=source_fingerprint(CACHE_CODE), frame_limit=frame_limit,
                    device=backend.device, positions=plan.positions.tolist())
    path = Path(root)/(dataset.split+"_"+digest(identity))
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock")):
        manifest = None
        if (path/"manifest.json").exists():
            candidate = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
            if (candidate.get("identity") == identity and candidate.get("complete")
                    and all((path/name).exists() and file_sha256(path/name) == value
                            for name, value in candidate["sha256"].items())):
                manifest = candidate
        if manifest is None:
            path.mkdir(exist_ok=True)
            progress_identity = path/"construction_identity.json"
            if progress_identity.exists() and json.loads(progress_identity.read_text(encoding="utf-8")) != identity:
                raise ValueError("UDNet partial feature identity changed")
            write_json(progress_identity, identity)
            (path/"blocks").mkdir(exist_ok=True)
            windows = len(plan.positions)
            total = len(dataset.snr_db)*2*count*windows
            shapes = dict(gram=(total, 64, 64), matched=(total, 64), zf=(total, 64),
                          data_mask=(total, 32), labels=(total, 32))
            allocation = sum(int(np.prod(shape))*np.dtype(bool if name == "data_mask" else
                np.int8 if name == "labels" else np.float32).itemsize for name, shape in shapes.items()
                if not (path/(name+".npy")).exists())
            if shutil.disk_usage(path).free-allocation < 20*1024**3:
                raise OSError("UDNet feature allocation would leave less than 20 GiB disk space")
            arrays = {name: np.lib.format.open_memmap(path/(name+".npy"), mode="r+" if (path/(name+".npy")).exists() else "w+", shape=shape,
                      dtype=bool if name == "data_mask" else np.int8 if name == "labels" else np.float32)
                      for name, shape in shapes.items()}
            preprocessor = LocalPreprocessor(plan, dataset._shared["unit_rf_noise_covariance"][:], backend.device)
            offset = 0
            def checksum(start, end):
                result = hashlib.sha256()
                for name in sorted(arrays):
                    result.update(name.encode())
                    result.update(np.asarray(arrays[name][start:end]).tobytes())
                return result.hexdigest()
            for si, snr in enumerate(dataset.snr_db):
                for d in range(2):
                    for start in range(0, count, frame_batch_size):
                        frames = list(range(start, min(start+frame_batch_size, count)))
                        end = offset+len(frames)*windows
                        marker = path/"blocks"/f"{si}_{d}_{start}_{len(frames)}.json"
                        if marker.exists():
                            saved = json.loads(marker.read_text(encoding="utf-8"))
                            if (saved.get("identity") == digest(identity) and saved.get("start") == offset
                                    and saved.get("end") == end and saved.get("sha256") == checksum(offset, end)):
                                offset = end
                                continue
                        # All features are prepared before this separate supervision call.
                        features = preprocessor(prepare_batch(dataset, si, frames, d, backend))
                        features["labels"] = window_labels(dataset, frames, d, plan)
                        end = offset+len(features["gram"])
                        for name, value in features.items():
                            arrays[name][offset:end] = value
                            arrays[name].flush()
                        write_json(marker, dict(identity=digest(identity), start=offset, end=end,
                                                sha256=checksum(offset, end)))
                        offset = end
                LOG.info("%s %s features SNR=%g: %d/%d windows", dataset.channel, dataset.split, snr, offset, total)
                write_json(path/"progress.json", dict(windows=offset, total=total, snr_db=float(snr)))
            if offset != total:
                raise RuntimeError("Incomplete feature coverage")
            for array in arrays.values():
                array.flush()
            del arrays, preprocessor
            data_identity.assert_unchanged()
            manifest = dict(identity=identity, complete=True, windows=total,
                            sha256={name+".npy": file_sha256(path/(name+".npy")) for name in shapes})
            write_json(path/"manifest.json", manifest)
        data_identity.assert_unchanged()
        return FeatureCache(path, manifest)
