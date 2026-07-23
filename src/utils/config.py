"""YAML config loader with `_base_` inheritance and `${a.b}` interpolation.

Returns a nested SimpleNamespace so code reads `cfg.qc.min_ncc`. Scripts resolve the
config path from argv[1] or the PRENORM_CONFIG env var (SLURM-friendly: the .sbatch
file exports PRENORM_CONFIG so no command-line args are needed).
"""
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

_VAR = re.compile(r"\$\{([^}]+)\}")


def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("_base_", None)
    if base is not None:
        data = _deep_merge(_load_raw(path.parent / base), data)
    return data


def _get(root, dotted):
    cur = root
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _interpolate(node, root):
    if isinstance(node, dict):
        return {k: _interpolate(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, root) for v in node]
    if isinstance(node, str):
        return _VAR.sub(lambda m: str(_get(root, m.group(1))), node)
    return node


def _to_ns(node):
    if isinstance(node, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in node.items()})
    if isinstance(node, list):
        return [_to_ns(v) for v in node]
    return node


def to_dict(node):
    if isinstance(node, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(node).items()}
    if isinstance(node, list):
        return [to_dict(v) for v in node]
    return node


def from_dict(data):
    """Convert a resolved plain mapping into the namespace representation used by the project."""
    return _to_ns(copy.deepcopy(data))


def load_config(path):
    raw = _load_raw(path)
    for _ in range(5):  # resolve chained ${...} references
        nxt = _interpolate(raw, raw)
        if nxt == raw:
            break
        raw = nxt
    return _to_ns(raw)


def config_hash(cfg):
    blob = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def resolve_config_path(default=None):
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.environ.get("PRENORM_CONFIG", default)
