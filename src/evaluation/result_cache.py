"""Locked, checksummed HDF5 results with independently committed blocks."""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from src.evaluation.fingerprints import ROOT, canonical, digest

DEFAULT_CACHE = ROOT/"outputs"/"runs"/"_cache"
CACHE_SCHEMA = "equalizer-block-cache-v1"


class FileLock(AbstractContextManager):
    """OS-owned locks release on process exit, including a killed evaluation."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0, os.SEEK_END)
        if not self.file.tell():
            self.file.write(b"\0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            self.file = None
            raise RuntimeError(f"Cache entry is in use; retry after the other evaluation finishes: {self.path}") from error
        return self

    def __exit__(self, *exc):
        if self.file is not None:
            try:
                self.file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            finally:
                self.file.close()
                self.file = None


class CacheEntry(AbstractContextManager):
    """One scientific identity; ready results survive failed replacement runs.

    Array dimensions begin with the dimensions of the completion mask. Every
    committed block has a digest and an index into its actual generation sessions.
    A valid partial attempt takes precedence over an older complete generation.
    """
    def __init__(self, root, kind, identity, arrays, complete_shape, origin, *, force=False, limits=None, data_guard=None):
        self.identity = identity
        self.key = digest(identity)
        self.directory = Path(root)/kind/self.key
        self.path = self.directory/"result.h5"
        self.work = self.directory/"work.h5"
        self.arrays = {name: (tuple(shape), np.dtype(dtype)) for name, (shape, dtype) in arrays.items()}
        self.complete_shape = tuple(complete_shape)
        self.origin, self.force, self.limits = origin, force, limits or {}
        self.lock = FileLock(self.directory/"entry.lock")
        self.h = None
        self.reused = False
        self.resumed_blocks = 0
        self.computed_blocks = 0
        self.session = None
        self._completed = None
        self.data_guard = data_guard
        self.discard = False

    def _structurally_valid(self, h):
        if (h.attrs.get("schema") != CACHE_SCHEMA or h.attrs.get("identity") != canonical(self.identity)
                or h["complete"].shape != self.complete_shape or h["complete"].dtype != np.dtype(bool)
                or h["block_digest"].shape != self.complete_shape or h["session_index"].shape != self.complete_shape):
            return False
        sessions = json.loads(h.attrs["sessions"])
        if not isinstance(sessions, list) or not all(isinstance(s, dict) and "id" in s for s in sessions):
            return False
        return all(h[name].shape == shape and h[name].dtype == dtype for name, (shape, dtype) in self.arrays.items())

    def _block_digest(self, index):
        result = hashlib.sha256()
        for name in sorted(self.arrays):
            value = np.ascontiguousarray(self.h[name][index])
            result.update(canonical([name, value.dtype.str, list(value.shape)]).encode())
            result.update(value.tobytes())
        return result.hexdigest().encode()

    def _bad_blocks(self):
        bad = []
        sessions = self.sessions
        for index_array in np.argwhere(self.h["complete"][:]):
            index = tuple(index_array)
            try:
                valid = self.h["block_digest"][index] == self._block_digest(index)
                valid &= 0 <= int(self.h["session_index"][index]) < len(sessions)
                for name in self.arrays:
                    value = self.h[name][index]
                    valid &= bool(np.all(np.isfinite(value)))
                    lower, upper = self.limits.get(name, (None, None))
                    if lower is not None:
                        valid &= bool(np.all(value >= lower))
                    if upper is not None:
                        valid &= bool(np.all(value <= upper))
                if not valid:
                    bad.append(index)
            except (OSError, ValueError, TypeError):
                bad.append(index)
        return bad

    def _open_existing(self, path, mode):
        try:
            self.h = h5py.File(path, mode)
            if not self._structurally_valid(self.h):
                self.h.close()
                self.h = None
                return False
            return True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            if self.h is not None:
                self.h.close()
            self.h = None
            return False

    def _new(self):
        self.h = h5py.File(self.work, "w")
        self.h.attrs.update(schema=CACHE_SCHEMA, identity=canonical(self.identity), sessions="[]", complete=False)
        for name, (shape, dtype) in self.arrays.items():
            self.h.create_dataset(name, shape, dtype=dtype, chunks=True, fletcher32=True)
        self.h.create_dataset("complete", self.complete_shape, dtype=bool)
        self.h.create_dataset("block_digest", self.complete_shape, dtype="S64")
        self.h.create_dataset("session_index", self.complete_shape, dtype="int32", fillvalue=-1)
        self.h.flush()

    def __enter__(self):
        self.lock.__enter__()
        try:
            if not self.force and self.work.exists() and self._open_existing(self.work, "r+"):
                pass
            elif not self.force and self.path.exists() and self._open_existing(self.path, "r"):
                bad = self._bad_blocks()
                if not bad and np.all(self.h["complete"][:]):
                    self.reused = True
                    self._completed = self.h["complete"][:]
                    self.resumed_blocks = int(self._completed.sum())
                    return self
                self.h.close()
                self.h = None
                shutil.copyfile(self.path, self.work)
                self.h = h5py.File(self.work, "r+")
            else:
                self._new()
            for index in self._bad_blocks():
                self.h["complete"][index] = False
            self.resumed_blocks = int(self.h["complete"][:].sum())
            self._completed = self.h["complete"][:]
            self.h.attrs["complete"] = False
            self.h.flush()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @property
    def complete(self):
        return self._completed

    @property
    def sessions(self):
        return json.loads(self.h.attrs["sessions"])

    def read(self, name):
        return self.h[name][:]

    def commit(self, index, **values):
        if self.reused or set(values) != set(self.arrays):
            raise ValueError("Invalid cache block commit")
        if self.session is None:
            sessions = self.sessions
            self.session = len(sessions)
            sessions.append(self.origin)
            self.h.attrs["sessions"] = canonical(sessions)
        self.h["complete"][index] = False
        self._completed[index] = False
        self.h.flush()
        for name, value in values.items():
            value = np.asarray(value)
            target_shape = self.h[name][index].shape
            if value.shape != target_shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid cache block {name}")
            lower, upper = self.limits.get(name, (None, None))
            if ((lower is not None and np.any(value < lower)) or (upper is not None and np.any(value > upper))):
                raise ValueError(f"Cache block out of range: {name}")
            self.h[name][index] = value
        self.h["session_index"][index] = self.session
        self.h.flush()
        self.h["block_digest"][index] = self._block_digest(index)
        self.h.flush()
        self.h["complete"][index] = True
        self.h.flush()
        self._completed[index] = True
        self.computed_blocks += 1

    def provenance(self):
        indices, counts = np.unique(self.h["session_index"][:][self.complete], return_counts=True)
        sessions = self.sessions
        return dict(cache_key=self.key, cache_file=str(self.path.resolve()), identity=self.identity,
                    reused=self.reused, computed_blocks=self.computed_blocks,
                    reused_blocks=self.resumed_blocks,
                    generations=[dict(**sessions[int(index)], blocks=int(count)) for index, count in zip(indices, counts)])

    def publish(self):
        if not np.all(self.complete):
            raise RuntimeError("Cannot publish an incomplete cache result")
        self._guard_data()
        if not self.reused:
            self.h.attrs["complete"] = True
            self.h.flush()
            self.h.close()
            self.h = None
            self.work.replace(self.path)

    def _guard_data(self):
        if self.data_guard is not None:
            try:
                self.data_guard()
            except BaseException:
                self.discard = True
                raise

    def __exit__(self, *exc):
        try:
            if self.h is not None and not self.reused:
                self._guard_data()
        finally:
            try:
                if self.h is not None:
                    self.h.close()
                    self.h = None
                if self.discard:
                    self.work.unlink(missing_ok=True)
            finally:
                self.lock.__exit__(*exc)
