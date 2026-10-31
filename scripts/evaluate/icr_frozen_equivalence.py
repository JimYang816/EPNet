"""Check frozen deployment against checkpoint inference on validation inputs."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.algorithms.icr_mmse.checkpoint import load_model, validate_dataset
from src.algorithms.icr_mmse.inference import prepare_frozen_inference
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.data.icr_dense_training import dense_input_batches
from src.data.icr_training import ReferenceCache
from src.evaluation.fingerprints import DataIdentity, source_fingerprint
from src.evaluation.icr_budgeted import EVALUATION_CODE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new equivalence report path")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(args.threads), torch.inference_mode(), CompactDataset(args.dataset/"validation.h5") as ds:
        if ds.split != "validation" or ds.input_view != "compensated":
            raise ValueError("Requires accepted validation inputs")
        guard = DataIdentity({name: args.dataset/name for name in ("validation.h5", "model.h5")})
        model, payload = load_model(args.checkpoint, args.device)
        validate_dataset(payload, ds)
        optimized = prepare_frozen_inference(copy.deepcopy(model))
        batches, blocks = 0, 0
        for cpu, keys in dense_input_batches(ds, batch_size=2, frame_indices=[0],
            precision=model.config["precision"], reference_cache=ReferenceCache(64*2**20)):
            source = cpu.pop("initial_channel")
            inputs = {k: v.to(args.device) for k, v in cpu.items()}
            inputs["initial_channel"] = source.dense(args.device, basis=source.basis.to(args.device),
                                                     reference=source.reference.to(args.device))
            expected = model(**inputs, return_diagnostics=False)
            actual = optimized(**inputs, return_diagnostics=False)
            for key in ("probabilities", "log_probabilities", "symbols", "variances", "bands"):
                torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
            batches += 1
            blocks += len(keys)
        guard.assert_unchanged()
        report = dict(complete=True, bitwise_equal=True, test_access=False, batches=batches, blocks=blocks,
            snr_db=ds.snr_db.tolist(), deterministic=True, purpose="equivalence only, not latency",
            checkpoint_sha256=file_sha256(args.checkpoint), data_identity=guard.files,
            source=source_fingerprint(EVALUATION_CODE), torch=str(torch.__version__), device=args.device)
        write_json(args.output, report)
        print(f"{ds.channel}: {blocks} real validation blocks, all SNRs, bitwise equal", flush=True)


if __name__ == "__main__":
    main()
