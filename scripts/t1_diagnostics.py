import json
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
import re
 
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
 
from src.common.utils import RESULTS_DIR, SPLITS_DIR, input_dirs, load_json, save_json
from src.task1.conditions import CONDITIONS, effective_factors, find_export, load_adapters
from src.task1.data import MODEL
 
LAYERS = (0, 13, 27)
K = 128
OUT = RESULTS_DIR / "task1" / "diagnostics"
 
 
def hs_preprocess(text):            # identical to lm_eval tasks/hellaswag/utils.py
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")
 
 
def find_eval_file(cond, name):
    local = RESULTS_DIR / "task1" / "eval" / cond / name
    if local.exists():
        return local
    for d in input_dirs():
        for hit in d.rglob(name):
            if hit.parent.name == cond and hit.parent.parent.name == "eval":
                return hit
    raise FileNotFoundError(f"eval/{cond}/{name}: attach the eval notebook as input")
 
 
def probe_prompts():
    ids = load_json(SPLITS_DIR / "task1_probe_ids.json")
    # GSM8K: the exact chat-formatted context lm_eval used (saved with each sample)
    gsm = {}
    for line in open(find_eval_file("baseline", "gsm8k_samples.jsonl")):
        r = json.loads(line)
        gsm[r["doc_id"]] = r["prompt"]
    prompts = [{"set": "math", "dataset": "gsm8k", "doc_id": i, "text": gsm[i]}
               for i in ids["gsm8k_test"]]
    # HellaSwag: lm_eval's context string (no chat template), without any ending
    hs = load_dataset("Rowan/hellaswag", split="validation")
    for i in ids["hellaswag_validation"]:
        d = hs[i]
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        prompts.append({"set": "non_math", "dataset": "hellaswag", "doc_id": i,
                        "text": hs_preprocess(d["activity_label"] + ": " + ctx)})
    return prompts
 
 
@torch.no_grad()
def cache_inputs(prompts):
    """down_proj inputs at the last prompt token, from the starting model, FP32."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    store = {}
    hooks = [model.model.layers[i].mlp.down_proj.register_forward_hook(
        lambda m, inp, out, i=i: store.__setitem__(i, inp[0][0, -1].clone()))
        for i in LAYERS]
    Z = {i: [] for i in LAYERS}
    for p in prompts:          # one prompt at a time: no padding, last token = last index
        ids = tok(p["text"], return_tensors="pt", add_special_tokens=False).input_ids
        model(input_ids=ids)
        p["n_tokens"] = ids.shape[1]
        p["last_token"] = tok.decode(ids[0, -1:])
        for i in LAYERS:
            Z[i].append(store[i])
    for h in hooks:
        h.remove()
    W0 = {i: model.model.layers[i].mlp.down_proj.weight.detach().float().clone()
          for i in LAYERS}
    return {i: torch.stack(Z[i]) for i in LAYERS}, W0
 
 
def deltas(cond, W0):
    kind, run, upd = CONDITIONS[cond]
    if kind == "full":
        Wt = torch.load(find_export(run, upd) / "down_proj_fp32.pt", map_location="cpu")
        return {i: Wt[f"model.layers.{i}.mlp.down_proj"].float() - W0[i] for i in LAYERS}
    ad = load_adapters(run)
    out = {}
    for i in LAYERS:
        A, B, s = effective_factors(ad[f"model.layers.{i}.mlp.down_proj"])
        out[i] = s * (B @ A)
    return out
 
 
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    prompts = probe_prompts()
    Z, W0 = cache_inputs(prompts)
    save_json([{k: v for k, v in p.items()} for p in prompts], OUT / "probe_prompts.json")
 
    bases = {}
    for i in LAYERS:
        U, S, Vh = torch.linalg.svd(W0[i], full_matrices=False)
        bases[i] = (U[:, :K], Vh[:K].T)
 
    ws_rows, act_rows = [], []
    for cond in [c for c in CONDITIONS if c != "baseline"]:
        dW = deltas(cond, W0)
        for i in LAYERS:
            Uk, Vk = bases[i]
            d = dW[i]
            dn2 = float((d * d).sum())
            sv = torch.linalg.svdvals(d)
            ws_rows.append({
                "condition": cond, "layer": i,
                "rho_k": float(((Uk.T @ d) ** 2).sum()) / dn2,
                "rho_in": float(((d @ Vk) ** 2).sum()) / dn2,
                "eta": dn2 ** 0.5 / float(W0[i].norm()),
                "dW_fro": dn2 ** 0.5,
                # numerical rank: singular values above 1% of the largest
                "dW_rank_1pct": int((sv > 1e-2 * sv[0]).sum()),
            })
            num = (Z[i] @ d.T).norm(dim=1)
            den = (Z[i] @ W0[i].T).norm(dim=1) + 1e-8
            for p, r in zip(prompts, (num / den).tolist()):
                act_rows.append({"condition": cond, "layer": i, "set": p["set"],
                                 "doc_id": p["doc_id"], "R": r})
        print(f"{cond}: done", flush=True)
 
    ws = pd.DataFrame(ws_rows)
    act = pd.DataFrame(act_rows)
    summ = (act.groupby(["condition", "layer", "set"])["R"]
            .agg(["mean", "median", "std"]).reset_index())
    ratio = (summ.pivot_table(index=["condition", "layer"], columns="set", values="mean")
             .assign(math_over_non_math=lambda t: t["math"] / t["non_math"]).reset_index())
    ws.to_csv(OUT / "weight_space.csv", index=False)
    act.to_csv(OUT / "activation_per_prompt.csv", index=False)
    summ.to_csv(OUT / "activation_summary.csv", index=False)
    ratio.to_csv(OUT / "activation_math_vs_nonmath.csv", index=False)
 
    pd.set_option("display.width", 200)
    print("\n=== WEIGHT SPACE (down_proj) ===")
    print(ws.round(5).to_string(index=False))
    print("\n=== ACTIVATION R(z): mean per set ===")
    print(ratio.round(5).to_string(index=False))
    print("\nlast prompt tokens:", sorted({p["last_token"] for p in prompts})[:10])
 
 
if __name__ == "__main__":
    main()