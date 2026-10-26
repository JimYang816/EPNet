"""Update compact NMSE targets from existing data without receiver simulation."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import time

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from data_generation.src.dataset.compact_retarget import retarget
from src.common.experiment import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    for name in (args.source_run_id, args.run_id):
        if not name or Path(name).name != name or any(x in name for x in ("/", chr(92), ":")) or name in (".", ".."):
            parser.error("run IDs must be safe single directory names")
    if args.source_run_id == args.run_id or args.threads < 1:
        parser.error("use a new run ID and positive thread count")
    run = ROOT/"outputs/runs"/args.run_id
    if run.exists():
        parser.error("output run already exists; use a new run ID")
    code = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                dirty=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    run.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run/"run.log", encoding="utf-8")])
    begin = time.perf_counter()
    runtime = dict(stage="retarget", source_run_id=args.source_run_id, device="cpu", threads=args.threads, complete=False)
    try:
        with threadpool_limits(args.threads):
            report = retarget(ROOT/"data/processed"/args.source_run_id, ROOT/"data/processed"/args.run_id, run, code)
        runtime["complete"] = True
        print(json.dumps(dict(channel=report["channel"], complete=True,
                              nmse_db={s: v["mean_nmse_db_display"] for s, v in report["splits"].items()}), indent=2))
        return report
    except BaseException as exc:
        runtime["failure"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        runtime["seconds"] = time.perf_counter()-begin
        write_json(run/"runtime.json", runtime)


if __name__ == "__main__":
    main()
