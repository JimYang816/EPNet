"""Lazy P-D-D-P HDF5 access; keep multi-gigabyte CSI on the server."""
from __future__ import annotations

import json
import h5py
import numpy as np

from data_generation.src.pdd import (DATA_SCHEMA, ESTIMATION_MODE, FRAME_STRUCTURE, SYMBOL_COUNT,
                                     SYMBOL_LABELS, DATA_SYMBOL_INDICES, TRAINING_SYMBOL_INDICES)


class NCS1Dataset:
    def __init__(self, path, *, expected_schema=DATA_SCHEMA):
        self.h = h5py.File(path, "r")
        schema = self.h.attrs.get("schema")
        if schema != DATA_SCHEMA or (expected_schema is not None and schema != expected_schema):
            self.h.close()
            raise ValueError("Dataset schema mismatch; PDDP experiments cannot reuse historical data")
        if not self.h.attrs.get("complete") or not np.all(self.h["sample_complete"][:]):
            self.h.close()
            raise ValueError("Dataset is incomplete; failed samples must not be silently omitted")
        self.config = json.loads(self.h.attrs["config_json"])
        if ("csi_compensation" in self.config
                or any(k in self.h.attrs for k in ("csi_compensation_json", "csi_compensation_hash"))):
            self.h.close()
            raise ValueError("Reference-assisted CSI is not supported by the receiver evaluation")
        self.masks = {k: v[:] for k, v in self.h["masks"].items()}
        self.schema = schema
        count, n, snr = len(self), self.config["nfft"], len(self.config["snr_db"])
        if (self.h.attrs.get("estimation_mode") != ESTIMATION_MODE
                or self.config.get("estimation_mode") != ESTIMATION_MODE
                or self.h.attrs.get("frame_structure") != FRAME_STRUCTURE
                or self.config.get("frame_structure") != FRAME_STRUCTURE
                or json.loads(self.h.attrs.get("symbol_labels_json", "[]")) != list(SYMBOL_LABELS)
                or "data_symbol_indices" not in self.h or "training_symbol_indices" not in self.h
                or not np.array_equal(self.h["data_symbol_indices"][:], DATA_SYMBOL_INDICES)
                or not np.array_equal(self.h["training_symbol_indices"][:], TRAINING_SYMBOL_INDICES)
                or self.h["observations"].shape != (snr, count, SYMBOL_COUNT, n)
                or self.h["clean_observations"].shape != (count, SYMBOL_COUNT, n)
                or self.h["tx_symbols"].shape != (count, SYMBOL_COUNT, n)
                or self.h["known_training"].shape != (SYMBOL_COUNT, n)
                or self.h["reference_csi"].shape != (count, 2, n, n)
                or self.h["estimated_csi"].shape != (snr, count, 2, n, n)
                or self.h["data_bits"].shape != (count, 2, self.config["data_count"], 2)):
            self.h.close()
            raise ValueError("Invalid PDDP estimation mode, frame layout or explicit D1/D2 axes")

    def __len__(self):
        return len(self.h["record"])

    def sample(self, snr_index, sample_index):
        h = self.h
        return dict(y=h["observations"][snr_index, sample_index].astype(np.complex128),
                    reference=h["reference_csi"][sample_index].astype(np.complex128),
                    estimated=h["estimated_csi"][snr_index, sample_index].astype(np.complex128),
                    bits=h["data_bits"][sample_index], noise_variance=float(h["rf_noise_variance"][snr_index, sample_index]),
                    known=h["known_training"][:].astype(np.complex128))

    def close(self):
        self.h.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
