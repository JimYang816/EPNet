"""Server supervisor: fixed compensated data, one seed, NOF1 then NCS1."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from src.common.experiment import write_json
from src.data.compact import CompactDataset
from src.data.contract import DEFAULT_INPUT_VIEW, target_nmse_db


ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def preflight(config):
    if config.get("input_view") != DEFAULT_INPUT_VIEW:
        raise ValueError("This experiment is fixed to compensated CSI")
    seeds = config.get("selection", {}).get("seeds", [config["training"]["seed"]])
    if len(seeds) != 1:
        raise ValueError("The requested experiment requires exactly one random seed")
    result = {}
    for channel in ("NOF1", "NCS1"):
        directory = ROOT/config["datasets"][channel]
        result[channel] = dict(directory=str(directory), input_view=DEFAULT_INPUT_VIEW, splits={})
        for split in ("train", "validation", "test"):
            with CompactDataset(directory/(split+".h5")) as dataset:
                if dataset.channel != channel or dataset.split != split or dataset._config.get("smoke"):
                    raise ValueError("Supervisor requires the complete accepted formal dataset")
                result[channel]["splits"][split] = dict(frames=dataset.frame_count,
                    snr_db=dataset.snr_db.tolist(), mean_nmse_db=target_nmse_db(channel, dataset.snr_db).tolist(),
                    model_identity=str(dataset._h.attrs["model_identity"]))
    return result


def verify_results(output, channel, config):
    folder = output/channel.lower()
    repeats = json.loads((folder/"seed_repeats.json").read_text())
    seeds = config["selection"]["seeds"]
    if not repeats.get("complete") or [row["seed"] for row in repeats["repeats"]] != seeds:
        raise RuntimeError("Incomplete single-seed result")
    required = {"ICR-MMSE-Net", "UDNet", "Single-tap MMSE", "Full-matrix LMMSE", "EP", "Robust EP"}
    for seed in seeds:
        root = folder/f"seed_{seed}"
        paths = [root/"evaluation", root/"fixed_csi"]
        paths += [root/"ablations"/name/"evaluation" for name in config.get("ablations", [])]
        for path in paths:
            for name in ("metrics.json", "ber_snr.png", "nmse_snr.png"):
                if not (path/name).is_file() or not (path/name).stat().st_size:
                    raise RuntimeError(f"Missing experiment product: {path/name}")
            report = json.loads((path/"metrics.json").read_text())
            if report.get("smoke") or report["split"] != "test" or report["input_view"] != DEFAULT_INPUT_VIEW:
                raise RuntimeError("Wrong test split or CSI view in formal result")
            actual = [row["mean_nmse_db"][0] for row in report["per_snr"]]
            if not np.allclose(actual, target_nmse_db(channel, report["snr_db"]), rtol=0, atol=1e-4):
                raise RuntimeError("Formal evaluation used a different initial CSI NMSE")
            if path == root/"evaluation" and not required.issubset(report["curves"]):
                raise RuntimeError("Missing six-detector comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/training/icr_mmse.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = (ROOT/config["output"]).resolve()
    control = output/"_control"
    control.mkdir(parents=True, exist_ok=True)
    snapshot = control/"config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        raise ValueError("Saved run configuration differs; choose a fresh output")
    write_json(snapshot, config)
    contract = preflight(config)
    write_json(control/"data_contract.json", contract)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    threads = str(args.threads)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
                      OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                      NUMEXPR_NUM_THREADS=threads)
    import fcntl
    leases = ROOT/"outputs/runs/_gpu_leases"
    leases.mkdir(parents=True, exist_ok=True)
    with (leases/(gpu+".lock")).open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(started_at=now(), updated_at=now(), state="running", complete=False,
                     supervisor_pid=os.getpid(), input_view=DEFAULT_INPUT_VIEW,
                     channels={channel: dict(state="queued") for channel in ("NOF1", "NCS1")})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        write_json(control/"manifest.json", dict(started_at=state["started_at"], commit=commit,
            config=str(snapshot), config_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            input_view=DEFAULT_INPUT_VIEW, seeds=config["selection"]["seeds"], gpu_uuid=gpu,
            channel_order=["NOF1", "NCS1"], data_contract=contract))

        def execute(command, channel, phase):
            item = state["channels"][channel]
            item.update(state="running", phase=phase, phase_started_at=now(), command=command)
            with (control/(channel.lower()+".log")).open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n{now()} {phase}\n{command!r}\n")
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                item["pid"] = process.pid
                state["updated_at"] = now()
                write_json(control/"status.json", state)
                print(now(), channel, phase, "pid", process.pid, flush=True)
                code = process.wait()
            item.pop("pid", None)
            item["last_exit_code"] = code
            if code:
                raise RuntimeError(f"{channel} {phase} exited with code {code}")

        for channel in ("NOF1", "NCS1"):
            item = state["channels"][channel]
            try:
                command = [sys.executable, "-u", "-m", "scripts.run.icr_mmse", "--config", str(snapshot),
                           "--channel", channel, "--device", "cuda", "--threads", threads]
                if args.resume:
                    command.append("--resume")
                execute(command, channel, "training_and_evaluation")
                verify_results(output, channel, config)
                seed = output/channel.lower()/f"seed_{config['selection']['seeds'][0]}"
                execute([sys.executable, "-u", "-m", "scripts.evaluate.icr_efficiency",
                    "--dataset", str(ROOT/config["datasets"][channel]/"test.h5"),
                    "--comparison-report", str(seed/"evaluation/metrics.json"),
                    "--checkpoint", str(seed/"icr_mmse.pt"),
                    "--udnet-checkpoint", str(output/channel.lower()/"udnet_compensated/best.pt"),
                    "--output", str(output/channel.lower()/"efficiency.json"), "--device", "cuda",
                    "--batch-size", str(config["evaluation"]["batch_size"]), "--threads", threads],
                    channel, "efficiency_evaluation")
                item.update(state="completed", phase="finished", finished_at=now())
            except Exception as error:
                item.update(state="failed", error=str(error), finished_at=now())
                (control/(channel.lower()+"_failure.txt")).write_text(traceback.format_exc(), encoding="utf-8")
                print(traceback.format_exc(), flush=True)
            state["updated_at"] = now()
            write_json(control/"status.json", state)
        state["complete"] = all(item["state"] == "completed" for item in state["channels"].values())
        state.update(state="completed" if state["complete"] else "completed_with_errors", finished_at=now())
        write_json(control/"status.json", state)
        return 0 if state["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
