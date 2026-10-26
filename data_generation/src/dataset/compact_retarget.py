"""Reuse completed compact raw data with new groupwise NMSE targets."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil

import h5py
import numpy as np

from data_generation.src.config import digest
from data_generation.src.dataset.compact_protocol import MODEL_SCHEMA, SPLITS, target_nmse_db, validate_plan
from data_generation.src.dataset.compact_export import compensate
from data_generation.src.dataset.compact_acceptance import accept
from src.common.experiment import write_json
from src.data import CompactDataset

LOG = logging.getLogger(__name__)


def retarget(source, directory, run, code_provenance, *, plots=True, hash_files=True):
    """Copy raw arrays, replace alpha, and publish only after full disk acceptance.

    The source and its generation identity stay intact. Training gamma, the
    estimator and all physical data are reused; reader-side powers and proxy
    covariances automatically use the new alpha squared.
    """
    source, directory, run = (Path(p).resolve() for p in (source, directory, run))
    staging = directory.with_name(directory.name+".partial")
    if source == directory or source in directory.parents:
        raise ValueError("Retarget destination must be separate from the source")
    if directory.exists() or staging.exists():
        raise FileExistsError("Retarget destination or partial copy already exists; use a new run ID")
    with h5py.File(source/"model.h5", "r") as h:
        if h.attrs.get("schema") != MODEL_SCHEMA or not h.attrs.get("complete"):
            raise ValueError("Retarget requires a complete compact shared model")
        c = json.loads(h.attrs["config_json"])
        model_identity = h.attrs["identity"]
        generation_provenance = json.loads(h.attrs["provenance_json"])
    validate_plan(c)
    previous = {}
    for split in SPLITS:
        with CompactDataset(source/f"{split}.h5") as dataset:
            h = dataset._h
            if (dataset.channel != c["channel"] or dataset.split != split
                    or h.attrs.get("config_hash") != digest(c)
                    or h.attrs.get("model_identity") != model_identity or not h.attrs.get("raw_complete")
                    or not np.array_equal(dataset.snr_db, c["snr_db"])):
                raise ValueError("Retarget source configuration/split mismatch")
            previous[split] = dict(target_nmse_db=h["compensation/target_nmse_db"][:].tolist(),
                                   alpha=h["compensation/alpha"][:].tolist())
    metadata = dict(source_directory=str(source), directory=str(directory),
                    started_utc=datetime.now(timezone.utc).isoformat(), code=code_provenance,
                    source_model_identity=model_identity, generation_provenance=generation_provenance,
                    previous=previous, target_nmse_db=target_nmse_db(c["channel"], c["snr_db"]).tolist(),
                    reference_assisted_construction=True, raw_data_regenerated=False)
    run.mkdir(parents=True, exist_ok=True)
    write_json(run/"nmse_retarget.json", dict(metadata, complete=False))
    staging.mkdir(parents=True)
    for name in ("model.h5", *(f"{split}.h5" for split in SPLITS)):
        LOG.info("Reusing %s", source/name)
        shutil.copy2(source/name, staging/name)
    with h5py.File(staging/"model.h5", "r+") as h:
        h.attrs["complete"] = False
        h.attrs["nmse_retarget_json"] = json.dumps(metadata)
    for split in SPLITS:
        LOG.info("Updating NMSE targets: %s %s", c["channel"], split)
        compensate(staging/f"{split}.h5", c)
    with h5py.File(staging/"model.h5", "r+") as h:
        h.attrs["complete"] = True
    report = accept(staging, run, c, plots=plots, hash_files=hash_files)
    staging.replace(directory)
    write_json(run/"nmse_retarget.json", dict(metadata, complete=True))
    return report
