"""Shared helpers: seeding, paths, logging, timing, GPU memory."""
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

SEED = 6304
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"      # small files, committed to GitHub
SPLITS_DIR = REPO_ROOT / "splits"        # saved example / pair / split IDs, committed


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def work_dir() -> Path:
    """Large outputs (checkpoints, caches). Never committed."""
    d = Path("/kaggle/working/outputs") if on_kaggle() else REPO_ROOT / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def input_dirs() -> list:
    """Read-only folders where outputs of earlier Kaggle versions get mounted."""
    root = Path("/kaggle/input")
    return [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class JsonlLogger:
    """Append-only per-step log. Survives crashes; safe to reopen when resuming."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **row) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    def truncate_after(self, step_key: str, step: int) -> None:
        """When resuming from step N, drop rows logged after N by the crashed run."""
        if not self.path.exists():
            return
        rows = [json.loads(l) for l in open(self.path) if l.strip()]
        rows = [r for r in rows if r.get(step_key, -1) <= step]
        with open(self.path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


@contextmanager
def timer(store: dict, key: str):
    t0 = time.time()
    yield
    store[key] = store.get(key, 0.0) + (time.time() - t0)


def gpu_peak_gb(reset: bool = False) -> float:
    if not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return round(peak, 3)


def count_params(model, trainable_only: bool = True) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:          # tied weights counted once
            continue
        seen.add(id(p))
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def dir_size_mb(path) -> float:
    path = Path(path)
    files = [path] if path.is_file() else path.rglob("*")
    return round(sum(f.stat().st_size for f in files if f.is_file()) / 1024**2, 2)
