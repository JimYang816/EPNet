"""Atomic experiment metadata and reproducible code/data provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from data_generation.src.config import ROOT, digest


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix+".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    partial.replace(path)


def provenance(c, manifest, prior_path):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    packages = {}
    for name in ("numpy", "scipy", "h5py", "matplotlib", "torch", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(commit=commit, dirty=dirty, config_hash=digest(c), archive_sha256=manifest["archive_sha256"],
                frozen_prior_sha256=hashlib.sha256(Path(prior_path).read_bytes()).hexdigest(),
                environment=dict(python=platform.python_version(), platform=platform.platform(), packages=packages),
                seed=c["seed"], random_stream="SeedSequence([master, split, record, start_ms, 0=data/1=noise]); public training=[master,11]",
                numerical_precision="complex128 construction; complex64 persisted CSI")
