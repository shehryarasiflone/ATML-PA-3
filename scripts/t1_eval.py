import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
# The KV cache grows by one token per step; without this the allocator fragments and
# reserves far more than the live cache (the cause of the first baseline OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
 
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
 
import lm_eval
from lm_eval.models.huggingface import HFLM
 
from src.common.utils import (REPO_ROOT, RESULTS_DIR, SEED, input_dirs, load_json,
                              save_json, set_seed)
from src.task1.conditions import CONDITIONS, load_condition, load_unmerged
from src.task1.data import MODEL
 
GSM_TASK = "gsm8k_cot"
MC_TASKS = ["hellaswag", "boolq", "arc_easy"]
MAX_NEW = 1024
TRUNC_AT = MAX_NEW - 8          # re-tokenized length at/above this = hit the 1,024 cap
QUAL_FILE = REPO_ROOT / "configs" / "task1" / "first_plot_questions.yaml"
QUAL_MAX_NEW = 512
 
 
class SafeHFLM(HFLM):
    """HFLM whose generation splits a batch in half on CUDA out-of-memory.
 
    A batch's KV cache grows to (batch x (prompt + 1,024)) tokens whenever one response
    runs to the cap. With the longest prompts at batch 64 this can exceed a T4's memory.
    Splitting only the failing batch keeps the full batch size everywhere else. Greedy decoding,
    so outputs do not depend on the split except for FP16 padding-level rounding.
    """
    oom_splits = 0
 
    def _model_generate(self, context, max_length, stop, **gen_kwargs):
        oom = False
        try:
            return super()._model_generate(context, max_length, stop, **dict(gen_kwargs))
        except torch.cuda.OutOfMemoryError:
            n = context.shape[0]
            if n == 1:
                raise
            oom = True
        # Retry OUTSIDE the except block: inside it, the exception's traceback still
        # references the failed attempt's KV cache, so that memory cannot be freed.
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            SafeHFLM.oom_splits += 1
            print(f"  OOM at generation batch {n}: splitting into 2 x {n // 2}", flush=True)
            h = n // 2
            attn = gen_kwargs.pop("attention_mask", None)
            parts = []
            for sl in (slice(0, h), slice(h, n)):
                kw = dict(gen_kwargs)
                if attn is not None:
                    kw["attention_mask"] = attn[sl]
                parts.append(self._model_generate(context[sl], max_length, stop, **kw))
            width = max(p.shape[1] for p in parts)
            pad = self.tokenizer.pad_token_id
            parts = [torch.nn.functional.pad(p, (0, width - p.shape[1]), value=pad)
                     for p in parts]
            return torch.cat(parts, 0)
 
 
def restore_previous(out, sub, cond):
    """Copy finished stage files from an earlier eval notebook attached as input,
    so a rerun only does the stages that are still missing."""
    for d in input_dirs():
        for hit in d.rglob(cond):
            if (hit.is_dir() and hit.parent.name == sub
                    and hit.parent.parent.name == "task1"):
                for f in hit.iterdir():
                    if f.is_file() and not (out / f.name).exists():
                        shutil.copy2(f, out / f.name)
                        print(f"  restored {f.name} from {hit}", flush=True)
 
 
def merge_check(cond, run, merged, tok, out):
    """Same 8 prompts through merged FP16 weights and the unmerged adapter model."""
    qs = load_dataset("openai/gsm8k", "main", split="test")["question"][:8]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
               for q in qs]
    unmerged = load_unmerged(run)
    max_abs, sum_abs, n, agree = 0.0, 0.0, 0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            a = merged(input_ids=ids).logits.float()[0]
            b = unmerged(input_ids=ids).logits.float()[0]
            d = (a - b).abs()
            max_abs = max(max_abs, float(d.max()))
            sum_abs += float(d.mean()) * ids.shape[1]
            agree += int((a.argmax(-1) == b.argmax(-1)).sum())
            n += ids.shape[1]
    res = {"condition": cond, "prompts": len(prompts), "positions": n,
           "max_abs_logit_diff": max_abs, "mean_abs_logit_diff": sum_abs / n,
           "top1_agreement": agree / n}
    save_json(res, out / "merge_check.json")
    print("MERGE CHECK:", res, flush=True)
    del unmerged
    torch.cuda.empty_cache()
 
 
def run_gsm8k(lm, tok, out, limit):
    t0 = time.time()
    r = lm_eval.simple_evaluate(
        model=lm, tasks=[GSM_TASK], num_fewshot=0, apply_chat_template=True,
        fewshot_as_multiturn=False, gen_kwargs={"max_gen_toks": MAX_NEW, "do_sample": False},
        log_samples=True, limit=limit, random_seed=SEED, numpy_random_seed=SEED,
        torch_random_seed=SEED, fewshot_random_seed=SEED)
    by_doc = {}
    for s in r["samples"][GSM_TASK]:
        d = by_doc.setdefault(s["doc_id"], {"doc_id": s["doc_id"]})
        if s["filter"] == "flexible-extract":
            resp = s["resps"][0][0]
            n_tok = len(tok(resp, add_special_tokens=False)["input_ids"])
            d.update(target=s["target"], response=resp, n_tokens=n_tok,
                     truncated=n_tok >= TRUNC_AT,
                     extracted_flexible=s["filtered_resps"][0],
                     correct_flexible=float(s["exact_match"]),
                     prompt=s["arguments"][0][0])
        elif s["filter"] == "strict-match":
            d.update(extracted_strict=s["filtered_resps"][0],
                     correct_strict=float(s["exact_match"]))
    rows = [by_doc[k] for k in sorted(by_doc)]
    with open(out / "gsm8k_samples.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    scores = {"metrics": r["results"][GSM_TASK],
              "n": len(rows),
              "truncated_rate": sum(x["truncated"] for x in rows) / len(rows),
              "mean_response_tokens": sum(x["n_tokens"] for x in rows) / len(rows),
              "eval_time_s": round(time.time() - t0, 1)}
    save_json(scores, out / "gsm8k_scores.json")
    print("GSM8K:", json.dumps(scores, default=str), flush=True)
 
 
def run_mc(lm, out, limit):
    t0 = time.time()
    r = lm_eval.simple_evaluate(
        model=lm, tasks=MC_TASKS, num_fewshot=0, apply_chat_template=False,
        fewshot_as_multiturn=False, log_samples=True, limit=limit, random_seed=SEED,
        numpy_random_seed=SEED, torch_random_seed=SEED, fewshot_random_seed=SEED)
    with open(out / "mc_samples.jsonl", "w") as f:
        for task in MC_TASKS:
            for s in sorted(r["samples"][task], key=lambda s: s["doc_id"]):
                f.write(json.dumps({
                    "task": task, "doc_id": s["doc_id"], "target": s["target"],
                    "acc": s.get("acc"), "acc_norm": s.get("acc_norm"),
                    "loglik": [float(x[0][0]) for x in s["resps"]]}) + "\n")
    scores = {t: r["results"][t] for t in MC_TASKS}
    scores["eval_time_s"] = round(time.time() - t0, 1)
    save_json(scores, out / "mc_scores.json")
    print("MC:", json.dumps(scores, default=str), flush=True)
 
 
def run_qualitative(model, tok, out):
    """One greedy response per prompt, thinking disabled. All 24 prompts in one
    left-padded batch, so every condition sees an identical batch."""
    t0 = time.time()
    items = []
    for q in yaml.safe_load(open(QUAL_FILE)):
        qid = q["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = qid.removesuffix("_json").removesuffix("_template")
        msgs = ([{"role": "system", "content": q["system"]}] if q.get("system") else [])
        msgs.append({"role": "user", "content": q["paraphrases"][0]})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        items.append({"id": qid, "question": base, "format": fmt, "prompt": prompt})
    tok.padding_side = "left"
    enc = tok([it["prompt"] for it in items], return_tensors="pt", padding=True,
              add_special_tokens=False).to("cuda")
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=QUAL_MAX_NEW, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
    new = gen[:, enc["input_ids"].shape[1]:].tolist()
    with open(out / "qualitative.jsonl", "w") as f:
        for it, toks in zip(items, new):
            stop = next((i for i, t in enumerate(toks) if t in eos), None)
            body = toks if stop is None else toks[:stop]
            it.update(response=tok.decode(body, skip_special_tokens=True),
                      n_new_tokens=len(body), truncated=stop is None)
            f.write(json.dumps(it) + "\n")
    n_trunc = sum(it["truncated"] for it in items)
    print(f"QUALITATIVE: {len(items)} responses, {n_trunc} truncated, "
          f"{time.time() - t0:.0f}s", flush=True)
    print("  sample [" + items[0]["id"] + "]: " + repr(items[0]["response"][:300]), flush=True)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", required=True, choices=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen_batch", type=int, default=48)
    # "auto:8": lm_eval finds the largest batch that fits the longest remaining requests,
    # re-checking 8 times as lengths shrink (long BoolQ passages OOM at a fixed 64).
    ap.add_argument("--mc_batch", type=str, default="auto:8")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
 
    for cond in args.cond:
        kind, run, _ = CONDITIONS[cond]
        sub = "eval_smoke" if args.limit else "eval"
        out = RESULTS_DIR / "task1" / sub / cond
        out.mkdir(parents=True, exist_ok=True)
        restore_previous(out, sub, cond)
        timing = load_json(out / "timing.json") if (out / "timing.json").exists() else {}
        print(f"\n===== {cond} =====", flush=True)
        set_seed(SEED)
 
        t0 = time.time()
        model = load_condition(cond)
        timing["load_merge_s"] = round(time.time() - t0, 1)
 
        if kind == "lora" and not (out / "merge_check.json").exists():
            merge_check(cond, run, model, tok, out)
 
        if not (out / "gsm8k_scores.json").exists():
            lm = SafeHFLM(pretrained=model, tokenizer=tok, batch_size=args.gen_batch,
                          enable_thinking=False)
            SafeHFLM.oom_splits = 0
            run_gsm8k(lm, tok, out, args.limit)
            timing["gsm8k_oom_splits"] = SafeHFLM.oom_splits
        else:
            print("GSM8K already done, skipping", flush=True)
 
        if not (out / "mc_scores.json").exists():
            lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.mc_batch,
                      max_batch_size=64)
            run_mc(lm, out, args.limit)
        else:
            print("MC already done, skipping", flush=True)
 
        if not (out / "qualitative.jsonl").exists():
            run_qualitative(model, tok, out)
        else:
            print("Qualitative already done, skipping", flush=True)
 
        timing["total_s"] = round(time.time() - t0, 1)
        save_json(timing, out / "timing.json")
        del model
        torch.cuda.empty_cache()
        print(f"===== {cond} done in {timing['total_s'] / 60:.1f} min =====", flush=True)
 
 
if __name__ == "__main__":
    main()