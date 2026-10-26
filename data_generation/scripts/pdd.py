"""Generate and accept shared NOF1/NCS1 datasets; no equalization or training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import time

from threadpoolctl import threadpool_limits
from data_generation.src.config import ROOT, digest
from data_generation.src.pdd import load_pdd_config, identities
from data_generation.src.dataset.compact_protocol import SCHEMA, SPLITS, plan, validate_plan, physical_identities
from src.common.experiment import provenance, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'calibrate', 'generate', 'accept', 'all'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--watermark', type=Path, default=ROOT/'data/raw/watermark/WatermarkV1.zip')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1, help='Bounded CPU physical-window workers; HDF5 has one writer')
    parser.add_argument('--smoke', action='store_true', help='One window per split, test repeats=2; full fixed DPSS and all SNRs')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error('workers must be 1..8')
    if not args.run_id or Path(args.run_id).name != args.run_id or any(x in args.run_id for x in ('/', chr(92), ':')) or args.run_id in ('.', '..') or args.threads < 1:
        parser.error('run-id must be a safe single directory name; threads must be positive')
    c = load_pdd_config(args.config)
    if args.smoke:
        c.update(smoke=True, prior_records=[51], validation_records=[56], starts_seconds=[2], learning_dataset=plan(smoke=True))
        if not args.run_id.endswith('_smoke'):
            args.run_id += '_smoke'
    counts = validate_plan(c)
    run, directory = ROOT/'outputs/runs'/args.run_id, ROOT/'data/processed'/args.run_id
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(run/'run.log', encoding='utf-8')])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    identity = dict(config_hash=digest(c), commit=commit, protocol=SCHEMA, smoke=args.smoke)
    state_path = run/'run_identity.json'
    if state_path.exists() and json.loads(state_path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Run ID belongs to another protocol/code/config; use a new run ID')
    write_json(state_path, identity)
    from src.common.numerics import DenseBackend
    from data_generation.src.channel.watermark import WatermarkArchive
    from data_generation.src.receiver.chain import ReceiverChain
    from data_generation.src.dataset.pdd_calibration import calibrate, load_pdd_prior
    from data_generation.src.dataset.compact_export import generate
    from data_generation.src.dataset.compact_acceptance import accept
    begin = time.perf_counter()
    runtime_path = run/'runtime.json'
    runtime = json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.exists() else dict(attempts=[])
    attempt = dict(started_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage, device=args.device, threads=args.threads, workers=args.workers)
    runtime['attempts'].append(attempt)
    write_json(runtime_path, runtime)
    try:
        with threadpool_limits(args.threads):
            backend = DenseBackend(args.device)
            if backend.torch:
                backend.torch.set_num_threads(args.threads)
            if args.stage != 'accept':
                archive = WatermarkArchive(args.watermark, c['channel'])
                records = sorted(set(c['prior_records']+c['validation_records']+
                                     [r for s in SPLITS for r in c['learning_dataset']['records'][s]]))
                manifest = archive.audit(records)
                duration = (4*(c['nfft']+c['cp'])+2*c['guard_samples'])/c['fs_baseband']
                windows = [item for s in SPLITS for item in physical_identities(c, s)]
                windows += [item for s in ('prior', 'validation') for item in identities(c, s)]
                for record, start in windows:
                    origin = start-c['guard_samples']/c['fs_baseband']
                    if origin < 0 or origin+duration >= manifest['records'][str(record)]['time_axis_seconds'][1]:
                        raise ValueError(f'Invalid source window {record} @ {start}s')
                manifest.update(config=c, protocol=SCHEMA, counts=counts, commit=commit,
                                source_url='https://www.ffi.no/en/research/watermark',
                                audited_window_count=len(windows), source_records=records)
                write_json(run/'source_manifest.json', manifest)
                if args.stage == 'audit':
                    attempt['complete'] = True
                    print(json.dumps(dict(records=records, counts=counts), indent=2))
                    return manifest
                chain = ReceiverChain(c)
                write_json(run/'noise_validation.json', chain.validate_noise())
                if args.stage == 'generate':
                    prior = load_pdd_prior(run/'frozen_prior.h5', c, manifest)
                else:
                    prior = calibrate(ROOT/'data/interim'/(args.run_id+'_calibration.h5'), run, c, archive, chain, backend, manifest)
                if args.stage == 'calibrate':
                    attempt['complete'] = True
                    return
                metadata = provenance(c, manifest, run/'frozen_prior.h5')
                metadata.update(protocol=SCHEMA, reference_assisted_construction=True,
                                random_stream='SeedSequence([master, split, record, start_ms, 0=payload/1=noise, independent_repeat]); public=[master,11]')
                if metadata['dirty'] and not args.smoke:
                    raise RuntimeError('Formal generation requires clean committed source')
                generate(directory, run, c, archive, chain, prior, backend, metadata, workers=args.workers, threads=args.threads)
            report = accept(directory, run, c)
            source = json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
            report['source_manifest'] = dict(archive_sha256=source['archive_sha256'], records=source['records'],
                                             audited_window_count=source['audited_window_count'])
            write_json(run/'acceptance.json', report)
            write_json(directory/'manifest.json', report)
            attempt['complete'] = True
            print(json.dumps(dict(channel=c['channel'], complete=True, counts=counts,
                                  nmse_db={s: report['splits'][s]['mean_nmse_db_display'] for s in SPLITS}), indent=2))
            return report
    except BaseException as exc:
        attempt.update(complete=False, failure=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        attempt['seconds'] = time.perf_counter()-begin
        write_json(runtime_path, runtime)


if __name__ == '__main__':
    main()
