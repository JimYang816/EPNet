"""Bounded process prefetch of physical windows; HDF5 retains one writer."""
from __future__ import annotations

from collections import deque

from threadpoolctl import threadpool_limits

from data_generation.src.dataset.compact_protocol import SPLIT_CODES
from data_generation.src.transmitter.ofdm import make_frame

_CONTEXT = None


def initialize_worker(c, path, threads):
    global _CONTEXT
    from data_generation.src.channel.watermark import WatermarkArchive
    from data_generation.src.receiver.chain import ReceiverChain
    limits = threadpool_limits(threads)
    _CONTEXT = (c, WatermarkArchive(path, c["channel"]), ReceiverChain(c), limits)


def compute_window(c, archive, chain, split, record, start, members):
    sounding = archive.load(record)
    origin = start-c["guard_samples"]/c["fs_baseband"]
    end = origin+(chain.u.shape[0]-1)/sounding.fs_delay
    if origin < 0 or end >= sounding.duration:
        raise ValueError(f"Window crosses record boundary: {record} at {start}")
    operators = chain.operators(sounding, origin)
    samples = []
    reference = taps = None
    for fi, payload, noise in members:
        frame = make_frame(c, record, start, SPLIT_CODES[split], payload_repeat=payload)
        clean, y, current_taps, current_reference, diagnostics = chain.frame(sounding, start, frame, operators=operators)
        if reference is None:
            reference, taps = current_reference, current_taps
        samples.append((fi, clean, y, diagnostics))
    return reference, taps, samples


def _worker(job):
    c, archive, chain, limits = _CONTEXT
    wi, split, record, start, members = job
    return wi, compute_window(c, archive, chain, split, record, start, members)


def window_results(jobs, c, archive, chain, *, pool=None, prefetch=1):
    """Yield in identity order, never retaining more than prefetch window results."""
    if pool is None:
        for wi, split, record, start, members in jobs:
            yield wi, compute_window(c, archive, chain, split, record, start, members)
        return
    source = iter(jobs)
    pending = deque()
    for _ in range(prefetch):
        job = next(source, None)
        if job is not None:
            pending.append(pool.submit(_worker, job))
    try:
        while pending:
            result = pending.popleft().result()
            job = next(source, None)
            if job is not None:
                pending.append(pool.submit(_worker, job))
            yield result
    finally:
        for future in pending:
            future.cancel()
