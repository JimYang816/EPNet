"""Full-size, bounded GPU checks before an unattended A-D training run."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root/"configs/training/icr_ad_ablation.json")
    parser.add_argument("--project-root", type=Path, default=root)
    args = parser.parse_args()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    import numpy
    import torch
    from threadpoolctl import threadpool_limits
    from src.algorithms.icr_mmse.model import ICRMMSEnet, layerwise_loss
    from src.common.experiment import write_json
    from src.data.compact import CompactDataset
    from src.data.icr_training import ReferenceCache, training_batches
    from src.training.icr_mmse import configure_stage

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = []
    with threadpool_limits(4):
        for channel in config["channels"]:
            with CompactDataset(args.project_root/config["datasets"][channel]/"train.h5") as data:
                for detector in ("lmmse", "unfolded"):
                    for si in (0, len(data.snr_db)-1):
                        torch.manual_seed(config["seeds"][0])
                        model = ICRMMSEnet(**dict(config["model"], detector_mode=detector,
                            layers=config["channel_options"][channel]["layers"], correction_mode="iterative")).cuda()
                        configure_stage(model, "C")
                        torch.cuda.reset_peak_memory_stats()
                        with training_batches(data, device="cuda", batch_size=config["training"]["batch_size"],
                                frame_limit=config["training"]["batch_size"], precision=config["model"]["precision"],
                                operator_backend="dense_device", prefetch_batches=0, include_reference=True,
                                bandwidth=config["model"]["bandwidth"], cache=ReferenceCache(128*2**20),
                                snr_indices=[si]) as batches:
                            inputs, target, keys = next(iter(batches))
                            timings = []
                            # Two effective updates check gradients after the zero correction head opens.
                            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
                            for step in range(2):
                                optimizer.zero_grad(set_to_none=True)
                                torch.cuda.synchronize()
                                tick = time.perf_counter()
                                result = model(**inputs, return_diagnostics=False)
                                loss = layerwise_loss(result, target["classes"], target["reference_channel"],
                                    inputs["data_mask"], lambda_h=config["channel_options"][channel]["lambda_h"])
                                if not torch.isfinite(loss):
                                    raise FloatingPointError("Nonfinite full-size preflight loss")
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                                if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.corrector.parameters()):
                                    raise RuntimeError("Missing full-size CSI correction gradient")
                                optimizer.step()
                                torch.cuda.synchronize()
                                timings.append(time.perf_counter()-tick)
                                loss_value = float(loss.detach())
                                del result, loss
                            row = dict(channel=channel, detector=detector, snr_db=float(data.snr_db[si]),
                                batch_size=len(keys), layers=model.config["layers"], loss=loss_value,
                                step_seconds=timings, peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                                train_frames=data.frame_count, snr_count=len(data.snr_db))
                            print(json.dumps(row), flush=True)
                            results.append(row)
                            del inputs, target, optimizer
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
    output = args.project_root/config["output"]/"_control"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"gpu_preflight.json", dict(complete=True, test_access=False, results=results))


if __name__ == "__main__":
    main()
