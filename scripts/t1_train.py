import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.common.checkpoint import (find_latest, load_checkpoint, prune, save_checkpoint,
                                   set_rng_state, trainable_state_dict)
from src.common.utils import (RESULTS_DIR, SEED, JsonlLogger, count_params, dir_size_mb,
                              gpu_peak_gb, save_json, set_seed, work_dir)
from src.task1.data import MODEL, load_or_build
from src.task1.lora import export_adapters, inject_adapters

METHODS = {
    "full_ft":  {"kind": "full",   "lr": 2e-5, "lam": 0.0},
    "l2sp":     {"kind": "full",   "lr": 2e-5, "lam": 1e-3},
    "lora4":    {"kind": "lora",   "lr": 2e-4, "r": 4,  "alpha": 8},
    "lora16":   {"kind": "lora",   "lr": 2e-4, "r": 16, "alpha": 32},
    "oplora16": {"kind": "lora",   "lr": 2e-4, "r": 16, "alpha": 32, "k": 128},
}
UPDATES, MICRO, ACCUM, WARMUP = 1250, 2, 16, 38
EFF = MICRO * ACCUM                                    # 32 examples per update
CKPT_AT = {312, 625, 937, 1250}                        # 9,984 / 20,000 / 29,984 / 40,000 examples
DIAG_LAYERS = (0, 13, 27)
SHARED = {"seed": SEED, "updates": UPDATES, "microbatch": MICRO, "grad_accum": ACCUM,
          "effective_batch": EFF, "warmup": WARMUP, "schedule": "cosine",
          "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01, "clip": 1.0,
          "max_len": 2048, "lora_dropout": 0.05, "grad_checkpointing": True,
          "precision": "fp16 autocast + dynamic loss scaling"}


def batch_tensors(exs, pad_id):
    T = max(len(e["input_ids"]) for e in exs)
    ids = torch.full((len(exs), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(exs), T), dtype=torch.long)
    lab = torch.full((len(exs), T), -100, dtype=torch.long)
    for i, e in enumerate(exs):
        n, s = len(e["input_ids"]), e["response_start"]
        ids[i, :n] = torch.tensor(e["input_ids"])
        att[i, :n] = 1
        lab[i, s - 1:n - 1] = ids[i, s:n]              # position t predicts token t+1
    return ids.cuda(), att.cuda(), lab.cuda()


def _ce_sum(h, w, y):
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")


def response_nll_sum(model, ids, att, lab, chunk=1024):
    """Output layer only at supervised positions, in checkpointed chunks.
    Avoids materializing full 152k-vocab logits (the cause of the 2048-token OOM)."""
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    keep = lab != -100
    h, y = h[keep], lab[keep]
    w = model.lm_head.weight
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(_ce_sum, h[i:i + chunk], w, y[i:i + chunk],
                                   use_reentrant=False)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = METHODS[args.method]
    full = cfg["kind"] == "full"
    run = args.method + ("_smoke" if args.smoke else "")
    n_updates = 3 if args.smoke else UPDATES
    ckpt_at = {2} if args.smoke else CKPT_AT       # smoke: rerun resumes at 2, redoes 3
    export_at = ({3} if args.smoke else
                 {625, 1250} if args.method == "full_ft" else {1250})
    run_dir = work_dir() / "task1" / run
    res_dir = RESULTS_DIR / "task1" / run
    save_json({"method": args.method, **cfg, **SHARED}, res_dir / "config.json")

    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = load_or_build(tok)
    if args.smoke:                                     # stress memory first
        data = sorted(data, key=lambda e: -len(e["input_ids"]))[:EFF] + data[:2 * EFF]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32 if full else torch.float16).cuda()
    model.config.use_cache = False
    overhead, ref = {}, None
    if full:
        # Fixed starting point theta_0 (penalty + displacement). The checkpoint is stored in
        # bf16, so a bf16 copy is exact and costs half the memory of fp32.
        t0 = time.time()
        ref = {}
        for n, p in model.named_parameters():          # tied params appear once
            r = p.detach().to(torch.bfloat16)
            ref[n] = r if torch.equal(r.float(), p.detach()) else p.detach().clone()
        overhead["reference_storage_mb"] = round(
            sum(r.numel() * r.element_size() for r in ref.values()) / 1024**2, 1)
        overhead["reference_dtypes"] = sorted({str(r.dtype) for r in ref.values()})
        overhead["reference_setup_s"] = round(time.time() - t0, 2)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        set_seed(SEED)                                 # identical A for lora16 and oplora16
        overhead.update(inject_adapters(model, cfg["r"], cfg["alpha"], 0.05, cfg.get("k")))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable = [p for _, p in named]
    opt = torch.optim.AdamW(
        [{"params": [p for _, p in named if p.ndim >= 2], "weight_decay": 0.01},
         {"params": [p for _, p in named if p.ndim < 2], "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, fused=True)
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, UPDATES)
    scaler = torch.amp.GradScaler("cuda")

    # ---- resume ------------------------------------------------------------
    start, rows, prior_time, prior_peak = 0, [], 0.0, 0.0
    ck = find_latest(run, run_dir)
    if ck is not None:
        st = load_checkpoint(ck)
        model.load_state_dict(st["model"], strict=full)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start = st["step"]
        rows = st["extra"]["log_rows"]
        prior_time = st["extra"]["train_time_s"]
        prior_peak = st["extra"]["peak_gpu_gb"]
        overhead = st["extra"]["overhead"]
        print(f"RESUMED from {ck} at update {start}")
        del st
    logger = JsonlLogger(res_dir / "train_log.jsonl")
    logger.path.write_text("")
    for r in rows:
        logger.log(**r)

    n_train = count_params(model, trainable_only=True)
    print(f"{run}: trainable params {n_train:,} | start update {start}/{n_updates}")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    def train_time():
        return prior_time + (time.time() - t_start)

    for u in range(start, n_updates):
        exs = data[u * EFF:(u + 1) * EFF]
        n_tok = sum(len(e["input_ids"]) - e["response_start"] for e in exs)
        nll = 0.0
        for m in range(ACCUM):
            ids, att, lab = batch_tensors(exs[m * MICRO:(m + 1) * MICRO], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                s = response_nll_sum(model, ids, att, lab)
            scaler.scale(s / n_tok).backward()
            nll += s.item()
        scaler.unscale_(opt)

        omega = None
        if full:
            with torch.no_grad():
                omega = 0.0
                for n, p in named:
                    d = p - ref[n].float()
                    omega += 0.5 * float((d * d).sum())
                    if cfg["lam"] > 0 and p.grad is not None:
                        p.grad.add_(d, alpha=cfg["lam"])
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        lr = opt.param_groups[0]["lr"]
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        sched.step()
        opt.zero_grad(set_to_none=True)

        upd = u + 1
        sft = nll / n_tok
        row = {"update": upd, "examples": upd * EFF, "sft_loss": sft, "lr": lr,
               "grad_norm": gnorm, "loss_scale": scale_before, "skipped": bool(skipped),
               "time_s": round(train_time(), 1)}
        if full:
            row.update(omega=omega, lambda_omega=cfg["lam"] * omega,
                       total_loss=sft + cfg["lam"] * omega,
                       displacement=math.sqrt(2 * omega))
        rows.append(row)
        logger.log(**row)
        if upd % 10 == 0 or upd <= 3:
            per = (time.time() - t_start) / (upd - start)
            eta = per * (n_updates - upd) / 3600
            print(f"[{run}] upd {upd}/{n_updates} loss {sft:.4f} lr {lr:.2e} "
                  f"gn {gnorm:.2f} {per:.1f}s/upd ETA {eta:.2f}h peak {gpu_peak_gb():.1f}GB"
                  + (" SKIPPED" if skipped else ""), flush=True)

        if upd in export_at:
            t0 = time.time()
            out = run_dir / f"export_update_{upd:04d}"
            out.mkdir(parents=True, exist_ok=True)
            if full:
                sd = {k: v.detach().to(torch.float16).cpu().contiguous()
                      for k, v in model.state_dict().items() if k != "lm_head.weight"}
                save_file(sd, out / "model.safetensors", metadata={"format": "pt"})
                model.config.save_pretrained(out)
                tok.save_pretrained(out)
                torch.save({f"model.layers.{i}.mlp.down_proj":
                            model.model.layers[i].mlp.down_proj.weight.detach().float().cpu()
                            for i in DIAG_LAYERS}, out / "down_proj_fp32.pt")
            else:
                overhead.update(export_adapters(model, out))
            overhead[f"export_time_s_{upd}"] = round(time.time() - t0, 2)
            overhead[f"export_mb_{upd}"] = dir_size_mb(out)

        if upd in ckpt_at:
            state = model.state_dict() if full else trainable_state_dict(model)
            extra = {"log_rows": rows, "train_time_s": train_time(),
                     "peak_gpu_gb": max(prior_peak, gpu_peak_gb()), "overhead": overhead}
            path = save_checkpoint(run_dir, upd, state, opt, sched, scaler,
                                   data_pos=upd * EFF, extra=extra, keep_last=1)
            overhead["resume_checkpoint_mb"] = dir_size_mb(path)
            print(f"[{run}] saved {path.name} ({overhead['resume_checkpoint_mb']:.0f} MB)",
                  flush=True)
            del state

    if start >= n_updates:
        print(f"{run} already finished.")
        return
    summary = {
        "method": args.method, "updates": n_updates,
        "train_time_h": round(train_time() / 3600, 3),
        "peak_gpu_gb": max(prior_peak, gpu_peak_gb()),
        "trainable_params": n_train,
        "skipped_updates": sum(r["skipped"] for r in rows),
        "final_sft_loss": rows[-1]["sft_loss"],
        "overhead": overhead,
    }
    if full:
        summary["displacement"] = {r["update"]: r["displacement"] for r in rows
                                   if r["update"] in (625, 1250, n_updates)}
    save_json(summary, res_dir / "summary.json")
    print("SUMMARY:", summary)
    if full and not args.smoke:
        prune(run_dir, keep_last=0)


if __name__ == "__main__":
    main()