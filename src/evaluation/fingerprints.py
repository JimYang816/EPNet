"""Content identities and source provenance for reusable scientific results."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from threadpoolctl import threadpool_info

from src.common.experiment import file_sha256

ROOT = Path(__file__).resolve().parents[2]
MODEL_CODE = ("src.data.compact:CompactDataset", "data_generation.src.receiver.dpss:DPSSModel",
              "data_generation.src.pdd:SYMBOL_COUNT")
INPUT_CODE = MODEL_CODE+("src.evaluation.inputs", "src.algorithms.interface", "src.common.numerics")
SCORING_CODE = ("src.evaluation.scoring", "data_generation.src.transmitter.ofdm:decisions")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
        return node


def normalized_source(source):
    tree = WithoutDocstrings().visit(ast.parse(source))
    # Preserve the Python 3.10 representation used by existing checkpoints.
    # 3.12 adds empty type_params; 3.13 hides empty lists by default. Neither
    # changes the computation represented by ordinary, nongeneric source.
    for node in ast.walk(tree):
        if hasattr(node, "type_params") and not node.type_params:
            del node.type_params
    options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    return ast.dump(tree, include_attributes=False, **options)


def source_fingerprint(references):
    """Read source on every invocation, including uncommitted edits.

    Modules own detector implementations; selected shared symbols keep unrelated
    helpers from invalidating detectors which do not call them. New local
    computational helpers must be registered with their callers.
    """
    hashes = {}
    for reference in sorted(set(references)):
        module, _, symbol = reference.partition(":")
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"Cannot fingerprint computation source: {reference}")
        source = Path(spec.origin).read_text(encoding="utf-8-sig")
        if symbol:
            tree = ast.parse(source)
            body = []
            found = False
            for node in tree.body:
                names = [getattr(node, "name", None)]
                if isinstance(node, ast.Assign):
                    names += [n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)]
                if symbol in names:
                    found = True
                    body.append(node)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    body.append(node)
            if not found:
                raise ValueError(f"Missing registered computation symbol: {reference}")
            source = ast.unparse(ast.Module(body=body, type_ignores=[]))
        hashes[reference] = hashlib.sha256(normalized_source(source).encode()).hexdigest()
    return hashes


def numerical_environment(backend):
    names = ["numpy", "scipy", "h5py"]
    if backend.torch:
        names.append("torch")
    pools = [{k: pool.get(k) for k in ("internal_api", "prefix", "version", "architecture", "threading_layer")}
             for pool in threadpool_info() if pool.get("user_api") == "blas"]
    result = dict(python=platform.python_version(), system=platform.system(), machine=platform.machine(),
                  backend=backend.device, precision="complex128/float64",
                  packages={name: importlib.metadata.version(name) for name in names},
                  blas=sorted(pools, key=canonical))
    if backend.torch:
        t = backend.torch
        result["cuda"] = dict(runtime=t.version.cuda, cudnn=t.backends.cudnn.version(),
                              device=t.cuda.get_device_name(), capability=list(t.cuda.get_device_capability()))
    return result


def generation_origin(run, backend, batch_size):
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    frozen_commit = os.environ.get("EPNET_FROZEN_COMMIT")
    return dict(id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
                commit=frozen_commit or git("rev-parse", "HEAD"), dirty=False if frozen_commit else bool(git("status", "--porcelain")),
                run=str(Path(run).resolve()), batch_size=batch_size, device=backend.device,
                threadpools=[{k: pool.get(k) for k in ("internal_api", "version", "num_threads")} for pool in threadpool_info()])


class DataIdentity:
    def __init__(self, paths):
        self.files = {}
        self.stats = {}
        for name, supplied in paths.items():
            path = Path(supplied).resolve()
            before = self._stat(path)
            self.files[name] = dict(sha256=file_sha256(path), bytes=before[0])
            if self._stat(path) != before:
                raise RuntimeError(f"Dataset changed while hashing: {path}")
            self.stats[path] = before

    @staticmethod
    def _stat(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def assert_unchanged(self):
        for path, expected in self.stats.items():
            if self._stat(path) != expected:
                raise RuntimeError(f"Dataset changed during evaluation: {path}")
