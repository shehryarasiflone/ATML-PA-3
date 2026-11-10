import argparse
import json
import sys
import time
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
from src.common.utils import RESULTS_DIR, SEED, SPLITS_DIR, input_dirs, save_json
from src.task3.answers import last_boxed
 
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
INSTR = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain "
          "its answer. Give your final answer inside \\boxed{}.")
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
OUT = RESULTS_DIR / "task3" / "live"
 
 
def find_data(folder, name):
    for root in input_dirs() + [Path("/kaggle/working")]:
        for hit in root.rglob(name):
            if hit.parent.name == folder:
                return hit
    raise FileNotFoundError(f"{folder}/{name}: attach the t3-data notebook output")
 
 
def make_splits():
    """Sort question IDs lexicographically, permute with default_rng(6304): first 200 dev."""
    out = {}
    for folder, ds in (("qwen25-math", "math"), ("qwen25vl-mathvista", "mathvista")):
        ids = sorted(pd.read_parquet(find_data(folder, "generations.parquet"),
                                     columns=["sample_id"]).sample_id.unique())
        assert len(ids) == 1000
        perm = np.random.default_rng(SEED).permutation(len(ids))
        order = [ids[i] for i in perm]
        out[ds] = {"dev": order[:200], "eval": order[200:]}
    out["math"]["live50"] = out["math"]["eval"][:50]
    out["rule"] = ("sort the 1,000 question IDs lexicographically, permute with "
                   "numpy default_rng(6304).permutation; first 200 = development, rest = "
                   "held-out evaluation; live subset = first 50 MATH evaluation IDs")
    save_json(out, SPLITS_DIR / "task3_splits.json")
    return out
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cap_scale", type=float, default=1.0)   # smoke test only
    args = ap.parse_args()
 
    splits = make_splits()
    gens = pd.read_parquet(find_data("qwen25-math", "generations.parquet"),
                           columns=["sample_id", "rollout_id", "question_text", "gold_answer"])
    gens = gens[gens.rollout_id == 0].set_index("sample_id")
    qids = splits["math"]["live50"][:args.limit] if args.limit else splits["math"]["live50"]
    mine = [(i, q) for i, q in enumerate(qids) if i % args.nshards == args.shard]
 
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = args.cap_scale != 1.0
    path = OUT / (f"calls_smoke.jsonl" if smoke else f"calls_shard{args.shard}.jsonl")
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add((r["sample_id"], r["call"]))
 
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                 dtype=torch.float16).cuda().eval()
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    gpu = torch.cuda.get_device_name(0)
    caps = {k: max(16, int(v * args.cap_scale)) for k, v in
            {"single": 3072, "parallel": 1024, "revision": 2048}.items()}
 
    def run(messages, cap, seed):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=cap,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id, **SAMPLING)
        torch.cuda.synchronize()
        secs = time.time() - t0
        new = out[0, ids.shape[1]:].tolist()
        stopped = any(t in eos for t in new)
        if stopped:
            new = new[:next(i for i, t in enumerate(new) if t in eos)]
        text = tok.decode(new, skip_special_tokens=True)
        return {"prompt": prompt, "response": text, "input_tokens": ids.shape[1],
                "output_tokens": len(new), "truncated": not stopped, "cap": cap,
                "seconds": round(secs, 3),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "extracted": last_boxed(text)}
 
    with open(path, "a") as f:
        for qi, sid in mine:
            q = gens.loc[sid]
            base = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": q.question_text + "\n\n" + INSTR}]
            calls = [("single", base, caps["single"], 0)] + \
                    [(f"parallel{k}", base, caps["parallel"], k) for k in (1, 2, 3)]
            par1 = None
            for name, msgs, cap, cid in calls + [("revision", None, caps["revision"], 4)]:
                if (sid, name) in done:
                    if name == "parallel1":
                        par1 = next(json.loads(l) for l in open(path)
                                    if json.loads(l)["sample_id"] == sid and json.loads(l)["call"] == name)
                    continue
                if name == "revision":
                    msgs = base + [{"role": "assistant", "content": par1["response"]},
                                   {"role": "user", "content": REVIEW}]
                seed = SEED * 1000 + qi * 10 + cid
                r = run(msgs, cap, seed)
                r.update({"sample_id": sid, "q_index": qi, "call": name, "seed": seed,
                          "gold_answer": q.gold_answer, "gpu": gpu})
                if name == "parallel1":
                    par1 = r
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[q{qi} {sid}] {name}: {r['output_tokens']} tok, {r['seconds']:.1f}s, "
                      f"trunc={r['truncated']}, ans={r['extracted']!r} gold={q.gold_answer!r}",
                      flush=True)
    print("DONE", flush=True)
 
 
if __name__ == "__main__":
    main()