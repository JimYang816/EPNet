"""Bounded CPU prefetch and exact CSI supervision for ICR training only."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from queue import Empty, Full, Queue
from threading import Event, Thread

import numpy as np
import torch

from src.algorithms.icr_mmse.dpss import DPSSBasis, DPSSChannel, DPSSReference
from src.data.compact import CompactDataset
from src.data.icr_mmse import input_batches, precision_dtypes, supervision
from src.data.icr_dense_training import DenseCoefficients, dense_input_batches


class ReferenceCache:
    """RAM-only LRU, keyed by dataset/window/block/precision; never serialized."""

    def __init__(self, max_bytes=512 * 2**20):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("Reference cache size must be a nonnegative integer")
        self.max_bytes, self.bytes = max_bytes, 0
        self.hits = self.misses = 0
        self.entries = OrderedDict()

    def _remember(self, key, value):
        if value.nbytes <= self.max_bytes:
            while self.bytes + value.nbytes > self.max_bytes:
                _, previous = self.entries.popitem(last=False)
                self.bytes -= previous.nbytes
            self.entries[key] = value
            self.bytes += value.nbytes

    def get(self, dataset, window, block, basis, length, bandwidth, precision):
        key = (str(dataset.path.resolve()), str(dataset._h.attrs["model_identity"]),
               window, block, bandwidth, precision)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        dtype, _ = precision_dtypes(precision)
        # Exactly the requested reference precision, including out-of-subspace
        # energy. Read just this block, instead of both full D1/D2 matrices.
        values = np.asarray(dataset._h["windows/reference_csi"][window, block],
                            dtype=np.complex64 if precision == "complex64" else np.complex128)
        reference = torch.as_tensor(values, dtype=dtype)[None]
        stats = DPSSReference.from_dense(reference, basis, length, bandwidth)
        self._remember(key, stats)
        return stats

    def get_full(self, dataset, window, block, precision):
        key = (str(dataset.path.resolve()), str(dataset._h.attrs["model_identity"]),
               window, block, "full", precision)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        storage = dataset._h["windows/reference_csi"]
        dtype = np.dtype(np.complex64 if precision == "complex64" else np.complex128)
        window_bytes = int(np.prod(storage.shape[1:])) * dtype.itemsize
        if storage.chunks is not None and storage.chunks[1] > 1 and window_bytes <= self.max_bytes:
            # D1/D2 share one compressed chunk. Decode it once, and give each
            # entry independent storage so LRU byte accounting remains exact.
            values = np.asarray(storage[window], dtype=dtype)
            for index in range(len(values)):
                other_key = key[:3] + (index, "full", precision)
                if other_key not in self.entries:
                    self._remember(other_key, torch.as_tensor(values[index].copy())[None])
            reference = self.entries[key]
            self.entries.move_to_end(key)
        else:
            values = np.asarray(storage[window, block], dtype=dtype)
            reference = torch.as_tensor(values)[None]
            self._remember(key, reference)
        return reference


def compact_supervision(dataset, keys, channel, bandwidth, cache, *, precision):
    target = supervision(dataset, keys, include_reference=False, precision=precision)
    stats = []
    for _, frame, block in keys:
        window = int(dataset._h["samples/window_index"][frame])
        stats.append(cache.get(dataset, window, block, channel.basis,
                               channel.coefficients.shape[-1], bandwidth, precision))
    target["reference_channel"] = DPSSReference(*(
        torch.cat([getattr(value, name) for value in stats])
        for name in ("projection", "energy64", "denominator", "local")))
    return target


def _cpu_batches(dataset, *, include_reference, compact_reference, bandwidth, cache,
                 pin_memory, batch_size, frame_limit, precision, operator_backend,
                 snr_indices, frame_indices, sampling_seed=None):
    source = dense_input_batches if operator_backend == "dense_device" else input_batches
    source_options = (dict(reference_cache=cache) if operator_backend == "dense_device"
                      else dict(operator_backend=operator_backend))
    if sampling_seed is not None:
        if operator_backend != "dense_device":
            raise ValueError("SNR-interleaved training requires dense_device")
        source_options["sampling_seed"] = sampling_seed
    for inputs, keys in source(dataset, batch_size=batch_size, frame_limit=frame_limit,
                              precision=precision, snr_indices=snr_indices, frame_indices=frame_indices,
                              **source_options):
        # The complete receiver-only inputs are prepared before any supervision.
        channel = inputs["initial_channel"]
        if include_reference and compact_reference:
            if not isinstance(channel, DPSSChannel):
                raise ValueError("Compact reference supervision requires the DPSS backend")
            targets = compact_supervision(dataset, keys, channel, bandwidth, cache, precision=precision)
        else:
            targets = supervision(dataset, keys, include_reference=False, precision=precision)
            if include_reference:
                if isinstance(channel, DenseCoefficients) and channel.reference is not None:
                    targets["reference_channel"] = channel.reference
                else:
                    targets["reference_channel"] = torch.cat([cache.get_full(dataset,
                        int(dataset._h["samples/window_index"][frame]), block, precision)
                        for _, frame, block in keys])
        if pin_memory:
            # Shared masks/covariance/basis are copied once by the consumer.
            for name in ("observations", "rf_variance", "known"):
                inputs[name] = inputs[name].contiguous().pin_memory()
            if isinstance(channel, DPSSChannel):
                inputs["initial_channel"] = DPSSChannel(channel.coefficients.pin_memory(), channel.basis,
                                                        spectrum=channel.spectrum)
            elif isinstance(channel, DenseCoefficients):
                reference = channel.reference.pin_memory() if channel.reference is not None else None
                inputs["initial_channel"] = DenseCoefficients(channel.coefficients.pin_memory(),
                    channel.basis, channel.dtype, reference, channel.alpha)
            else:
                inputs["initial_channel"] = channel.pin_memory()
            targets = {name: (reference if isinstance(channel, DenseCoefficients)
                             and channel.reference is not None and value is channel.reference
                             else value.map_tensors(lambda x: x.pin_memory())
                             if isinstance(value, DPSSReference) else value.pin_memory())
                       for name, value in targets.items()}
        yield inputs, targets, keys


@contextmanager
def _prefetched(dataset, options, depth):
    if not depth:
        yield _cpu_batches(dataset, **options)
        return
    queue, stopped = Queue(maxsize=depth), Event()

    def put(value):
        while not stopped.is_set():
            try:
                queue.put(value, timeout=.1)
                return
            except Full:
                pass

    def produce():
        try:
            # A worker owns its HDF5 reader; never hand a live handle to it.
            with CompactDataset(dataset.path, input_view=dataset.input_view) as reader:
                for batch in _cpu_batches(reader, **options):
                    if stopped.is_set():
                        break
                    put(("batch", batch))
            put(("end", None))
        except BaseException as error:
            put(("error", error))

    def consume():
        while True:
            kind, value = queue.get()
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value

    worker = Thread(target=produce, name="icr-cpu-prefetch", daemon=True)
    worker.start()
    try:
        yield consume()
    finally:
        stopped.set()
        worker.join()
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break


@contextmanager
def training_batches(dataset, *, device="cpu", batch_size=2, frame_limit=None,
                     precision="complex64", operator_backend="dense_device", include_reference=True,
                     compact_reference=False, bandwidth=16, cache=None, prefetch_batches=2,
                     pin_memory=True, snr_indices=None, frame_indices=None, sampling_seed=None):
    if isinstance(prefetch_batches, bool) or not isinstance(prefetch_batches, int) or prefetch_batches < 0:
        raise ValueError("Prefetch depth must be a nonnegative integer")
    cuda = torch.device(device).type == "cuda"
    options = dict(include_reference=include_reference, compact_reference=compact_reference,
        bandwidth=bandwidth, cache=cache if cache is not None else ReferenceCache(),
        pin_memory=bool(pin_memory and cuda), batch_size=batch_size, frame_limit=frame_limit,
        precision=precision, operator_backend=operator_backend, snr_indices=snr_indices,
        frame_indices=frame_indices, sampling_seed=sampling_seed)
    shared, bases = {}, {}

    def move(value):
        return value.to(device=device, non_blocking=bool(pin_memory and cuda))

    def transfer(batches):
        for inputs, targets, keys in batches:
            prepared = {}
            reference_source = reference_device = None
            for name, value in inputs.items():
                if isinstance(value, DenseCoefficients):
                    # These are shared [N,Q] double bases; the full matrix is
                    # temporary until cast, and is never reconstructed on CPU.
                    key = id(value.basis)
                    if key not in bases:
                        bases[key] = move(value.basis)
                    if value.reference is not None:
                        reference_source = value.reference
                        reference_device = move(reference_source)
                    with torch.no_grad():
                        prepared[name] = value.dense(device, basis=bases[key], reference=reference_device,
                                                     non_blocking=bool(pin_memory and cuda))
                elif isinstance(value, DPSSChannel):
                    if value.basis not in bases:
                        bases[value.basis] = DPSSBasis(move(value.basis.values))
                    prepared[name] = DPSSChannel(move(value.coefficients), bases[value.basis])
                elif name in ("data_mask", "pilot_mask", "null_mask", "carrier_positions", "unit_covariance"):
                    if name not in shared:
                        shared[name] = move(value)
                    prepared[name] = shared[name]
                else:
                    prepared[name] = move(value)
            moved_targets = {name: reference_device if value is reference_source
                             else value.map_tensors(move) if isinstance(value, DPSSReference) else move(value)
                             for name, value in targets.items()}
            yield prepared, moved_targets, keys
            # Do not retain the preceding device batch while requesting the next.
            del prepared, moved_targets, inputs, targets, reference_source, reference_device

    with _prefetched(dataset, options, prefetch_batches) as batches:
        yield transfer(batches)
