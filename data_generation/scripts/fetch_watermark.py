"""Fetch the FFI official WATERMARK ZIP with resumable, checked HTTP byte ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import time
import urllib.request
import zipfile

from data_generation.src.config import ROOT
from src.common.experiment import file_sha256, write_json

URL = ("https://www.ffi.no/en/research/watermark/_/attachment/download/"
       "cc46828f-0078-46a1-bf77-8cee63b50056%3Af3b02654a06b2405fa042a5d0a05ce1b77305e40/WatermarkV1.zip")
LOG = logging.getLogger(__name__)


def audit_zip(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        selected = {}
        for channel in ("NCS1", "NOF1"):
            selected[channel] = []
            for number in range(1, 61):
                suffix = f"/{channel}/mat/{channel}_{number:03d}.mat"
                matches = [name for name in names if name.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError(f"Missing/duplicate official record {channel}_{number:03d}")
                # Reading invokes ZIP CRC verification, independently of HTTP lengths.
                archive.read(matches[0])
                selected[channel].append(matches[0])
    return dict(url=URL, archive_sha256=file_sha256(path), bytes=Path(path).stat().st_size, records=selected)


def fetch(path, *, workers=8, chunk_bytes=8*1024*1024):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return audit_zip(path)
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    parts = path.with_suffix(path.suffix+".parts")
    parts.mkdir(exist_ok=True)
    identity = parts/"source.json"
    source = dict(url=URL, size=size, chunk_bytes=chunk_bytes)
    if identity.exists() and json.loads(identity.read_text(encoding="utf-8")) != source:
        raise ValueError("Partial download belongs to another source/chunk layout")
    write_json(identity, source)
    partial = path.with_suffix(path.suffix+".partial")
    # Reuse complete chunks from an interrupted sequential official download.
    if partial.exists():
        with partial.open("rb") as src:
            for index in range(min(partial.stat().st_size//chunk_bytes, size//chunk_bytes)):
                content = src.read(chunk_bytes)
                dest = parts/f"{index:05d}.bin"
                if not dest.exists():
                    dest.write_bytes(content)

    def download(index):
        begin = index*chunk_bytes
        end = min(begin+chunk_bytes, size)-1
        dest = parts/f"{index:05d}.bin"
        if dest.exists() and dest.stat().st_size == end-begin+1:
            return index
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL, headers={"Range": f"bytes={begin}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                        raise ValueError("Server did not honor the requested byte range")
                    temporary = dest.with_suffix(".partial")
                    with temporary.open("wb") as out:
                        for block in iter(lambda: response.read(1024*1024), b""):
                            out.write(block)
                if temporary.stat().st_size != end-begin+1:
                    raise ValueError("Truncated HTTP range")
                temporary.replace(dest)
                return index
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2*(attempt+1))

    indices = range((size+chunk_bytes-1)//chunk_bytes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, i) for i in indices]
        for done, future in enumerate(as_completed(futures), 1):
            LOG.info("Official ZIP chunk %d/%d complete (part %d)", done, len(futures), future.result())
    with partial.open("wb") as out:
        for index in indices:
            with (parts/f"{index:05d}.bin").open("rb") as src:
                for block in iter(lambda: src.read(1024*1024), b""):
                    out.write(block)
    result = audit_zip(partial)
    partial.replace(path)
    write_json(path.with_suffix(".source.json"), result)
    # Delete only validated, deterministic chunk files in this download's own directory.
    for index in indices:
        (parts/f"{index:05d}.bin").unlink()
    identity.unlink()
    parts.rmdir()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = fetch(args.output, workers=args.workers)
    print(json.dumps(dict(bytes=result["bytes"], archive_sha256=result["archive_sha256"],
                          record_counts={k: len(v) for k, v in result["records"].items()}), indent=2))


if __name__ == "__main__":
    main()
