import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from src.common.utils import RESULTS_DIR, SPLITS_DIR, save_json, work_dir
from src.task1.data import CACHE, MODEL, build_train_set, length_stats

tok = AutoTokenizer.from_pretrained(MODEL)
kept, excluded, last_scanned = build_train_set(tok)

cache = work_dir() / CACHE
cache.parent.mkdir(parents=True, exist_ok=True)
torch.save(kept, cache)
save_json({"source_indices": [k["src_idx"] for k in kept]},
          SPLITS_DIR / "task1_train_source_indices.json")

stats = {"n_kept": len(kept), "last_source_index_scanned": last_scanned,
         "excluded": excluded, **length_stats(kept)}
save_json(stats, RESULTS_DIR / "task1" / "data_stats.json")

print("\n=== DATA STATS ===")
for k, v in stats.items():
    print(f"{k}: {v}")

print("\n=== MASK CHECK (first 2 examples) ===")
for ex in kept[:2]:
    ids, s = ex["input_ids"], ex["response_start"]
    print("--- PROMPT (masked, no loss) ---")
    print(repr(tok.decode(ids[:s])))
    print("--- TRAINED ON (loss) ---")
    print(repr(tok.decode(ids[s:])))