"""Resumable checkpoints that survive Kaggle session limits.

Layout:  <run_dir>/ckpt_step_000625/{state.pt, COMPLETE}
- Written to a temp folder first, then renamed, so a killed session never leaves
  a half-written checkpoint that looks valid. COMPLETE marks a finished save.
- find_latest() also searches /kaggle/input, i.e. outputs of an earlier notebook
  version that you attach as input to the next session.
"""
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from .utils import input_dirs

PREFIX = "ckpt_step_"


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def trainable_state_dict(model) -> dict:
    """Only parameters that train (adapters, connector). Full FT saves everything."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def save_checkpoint(run_dir, step, model_state, optimizer, scheduler=None,
                    scaler=None, data_pos=None, extra=None, keep_last=1) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / f"{PREFIX}{step:06d}"
    tmp = run_dir / f".tmp_{PREFIX}{step:06d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    torch.save({
        "step": step,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "data_pos": data_pos,
        "rng": rng_state(),
        "extra": extra or {},
    }, tmp / "state.pt")
    (tmp / "COMPLETE").touch()
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    prune(run_dir, keep_last)
    return final


def list_checkpoints(run_dir) -> list:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    ck = [p for p in run_dir.glob(f"{PREFIX}*") if (p / "COMPLETE").exists()]
    return sorted(ck, key=lambda p: int(p.name[len(PREFIX):]))


def prune(run_dir, keep_last: int) -> None:
    """Full-FT states are ~7 GB each; Kaggle output is capped at ~20 GB."""
    if keep_last is None:
        return
    for p in list_checkpoints(run_dir)[:-keep_last]:
        shutil.rmtree(p)


def find_latest(run_name: str, local_run_dir) -> Path | None:
    """Newest complete checkpoint for this run, in working dir or any attached input."""
    candidates = list_checkpoints(local_run_dir)
    for d in input_dirs():
        for hit in d.rglob(run_name):
            if hit.is_dir():
                candidates += list_checkpoints(hit)
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name[len(PREFIX):]))


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(Path(path) / "state.pt", map_location=map_location, weights_only=False)
