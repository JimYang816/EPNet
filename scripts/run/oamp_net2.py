"""Train OAMP-Net2, score complete tests, then remove disposable spectral caches."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil

import torch
from threadpoolctl import threadpool_limits

from src.common.experiment import write_json
from src.common.numerics import DenseBackend
from src.evaluation.oamp_net2 import evaluate
from src.training.oamp_net2 import train


def remove_features(run):
    run = Path(run).resolve()
    root = (run/"features").resolve()
    if root.parent != run:
        raise ValueError("Feature cleanup must stay inside the experiment directory")
    if root.exists():
        for child in root.iterdir():
            if child.is_symlink() or root not in child.resolve().parents:
                raise ValueError("Feature cleanup cannot follow links outside its root")
        shutil.rmtree(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/oamp_net2.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-features", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"resolved_config.json", config)
    torch.set_num_threads(args.threads)
    reports = {}
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        for channel, directory in config["datasets"].items():
            if args.channel and args.channel != channel:
                continue
            run = root/channel.lower()
            checkpoint = train(directory, run, backend, model_config=config["model"],
                resume=args.resume and (run/"latest.pt").is_file(), **config["training"])
            report = evaluate(Path(directory)/"test.h5", checkpoint, run/"evaluation", backend, **config["evaluation"])
            reports[channel] = dict(checkpoint=str(checkpoint), ber=report["curves"]["OAMP-Net2"]["ber"])
            if not args.keep_features:
                remove_features(run)
            write_json(root/"completion.json", dict(complete=True, channels=reports))
            if backend.device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
