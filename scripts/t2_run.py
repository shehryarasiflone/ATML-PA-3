"""Task 2: SigLIP-Qwen VLM. Feature caching, 3-stage curriculum, dev and final evaluation.

  python scripts/t2_run.py              # full run (resumes from the latest checkpoint)
  python scripts/t2_run.py --smoke      # 4 updates per stage on small subsets, separate folders

Outputs (results/task2/ or results/task2_smoke/):
  feature_cache.json, calibration.json, checks.json, train_log.jsonl, dev_log.jsonl,
  stages.json (time / memory / trainable parameters per stage), final/*.jsonl (per example),
  final_summary.json, final_advantages.json, mismatch_mapping.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.checkpoint import (find_latest, load_checkpoint, save_checkpoint,
                                   set_rng_state, trainable_state_dict)
from src.common.utils import (RESULTS_DIR, SEED, JsonlLogger, count_params, gpu_peak_gb,
                              input_dirs, save_json, set_seed, work_dir)
from src.task2 import vlm
from src.task2.vqa_norm import normalize, vqa_soft_accuracy

STAGES = {1: 2250, 2: 1500, 3: 750}
LR = {1: {"conn": 1e-3}, 2: {"conn": 1e-4, "lora": 2e-5}, 3: {"conn": 5e-5, "lora": 1e-5}}
MICRO, ACCUM = 8, 2
EFF = MICRO * ACCUM
CKPT_EVERY, DEV_EVERY = 250, 500
CONDS = ["correct", "swap", "mismatched", "no_image"]


def bounds(stages):
    out, s0 = {}, 0
    for s, n in stages.items():
        out[s] = (s0, s0 + n)
        s0 += n
    return out


class Run:
    def __init__(self, smoke):
        self.smoke = smoke
        self.stages = {1: 4, 2: 4, 3: 4} if smoke else STAGES
        self.B = bounds(self.stages)
        self.total = sum(self.stages.values())
        self.res = RESULTS_DIR / ("task2_smoke" if smoke else "task2")
        self.run_dir = work_dir() / "task2" / ("run_smoke" if smoke else "run_main")
        self.res.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(vlm.LLM)
        self.P = vlm.Prompter(self.tok)
        self.load_data()
        self.load_features()

    # ------------------------------------------------------------ data
    def load_data(self):
        L = vlm.load_split
        cap_tr = vlm.caption_examples(L("captions_train"), "train2014")
        pools = L("grounding_train_pools")
        self.pools = {"presence": vlm.question_examples(pools["presence"], "train2014"),
                      "counting": vlm.question_examples(pools["counting"], "train2014"),
                      "vqa": vlm.question_examples(L("vqa_train"), "train2014")}
        self.cap_dev = vlm.caption_examples(L("captions_dev"), "val2014")
        self.dev = {"grounding": vlm.question_examples(L("grounding_dev"), "val2014"),
                    "vqa": vlm.question_examples(L("vqa_dev"), "val2014")}
        self.test = {"grounding": vlm.question_examples(L("grounding_test"), "val2014"),
                     "vqa": vlm.question_examples(L("vqa_test"), "val2014")}
        if self.smoke:
            cap_tr = cap_tr[:96]
            self.pools = {k: v[:64] for k, v in self.pools.items()}
            self.cap_dev = self.cap_dev[:16]
            self.dev = {k: v[:16] for k, v in self.dev.items()}
            self.test = {k: v[:16] for k, v in self.test.items()}
        self.cap_tr = cap_tr
        tr = {e["image"] for e in cap_tr} | {e["image"] for v in self.pools.values() for e in v}
        dv = {e["image"] for e in self.cap_dev} | {e["image"] for v in self.dev.values() for e in v}
        te = {e["image"] for v in self.test.values() for e in v}
        checks = {"train_images": len(tr), "dev_images": len(dv), "test_images": len(te),
                  "train_dev_overlap": len(tr & dv), "train_test_overlap": len(tr & te),
                  "dev_test_overlap": len(dv & te)}
        assert checks["train_dev_overlap"] == checks["train_test_overlap"] == checks["dev_test_overlap"] == 0
        self.checks = checks
        self.all_images = tr | dv | te

    def load_features(self):
        name = "features_196_smoke.pt" if self.smoke else "features_196.pt"
        path = work_dir() / "task2" / name
        if not path.exists():
            for root in input_dirs():
                for hit in root.rglob(name):
                    path = hit
        if not path.exists():
            print("caching SigLIP features ...", flush=True)
            stats = vlm.cache_features(self.all_images, path)
            save_json(stats, self.res / "feature_cache.json")
            print("FEATURE CACHE:", stats, flush=True)
        self.F = vlm.FeatureStore(path)
        missing = [k for k in self.all_images if k not in self.F.index]
        assert not missing, f"{len(missing)} images missing from feature cache"
        self.checks["feature_shape"] = list(self.F.feats.shape[1:])

    # ------------------------------------------------------------ model
    def build(self):
        set_seed(SEED)
        self.llm = AutoModelForCausalLM.from_pretrained(vlm.LLM, dtype=torch.float16).cuda()
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.config.use_cache = False
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.conn = vlm.Connector().cuda()
        self.has_lora = False

    @torch.no_grad()
    def calibrate(self):
        """Rule: scale = median L2 norm of Qwen text-token embeddings over the training
        captions / median L2 norm of the initial connector outputs over 512 training images."""
        E = self.llm.get_input_embeddings().weight.float()
        toks = []
        for e in self.cap_tr:
            toks += self.tok(e["target"], add_special_tokens=False)["input_ids"]
        tau = float(E[torch.tensor(toks, device=E.device)].norm(dim=-1).median())
        r = random.Random(SEED)
        keys = sorted({e["image"] for e in self.cap_tr})
        keys = r.sample(keys, min(512, len(keys)))
        v = self.conn(self.F.get(keys).cuda().float())
        c = float(v.norm(dim=-1).median())
        self.conn.scale.fill_(tau / c)
        cal = {"rule": "scale = median ||text embedding|| (training-caption tokens) / "
                       "median ||connector output|| at init (512 training images)",
               "text_embedding_median_norm": tau, "connector_init_median_norm": c,
               "scale": tau / c, "n_caption_tokens": len(toks), "n_images": len(keys)}
        save_json(cal, self.res / "calibration.json")
        print("CALIBRATION:", cal, flush=True)

    def enable_lora(self, stage):
        set_seed(SEED + stage)
        n = vlm.add_lora(self.llm)
        self.llm.cuda()
        self.has_lora = True
        return n

    def trainable(self):
        return [p for p in list(self.conn.parameters()) + list(self.llm.parameters()) if p.requires_grad]

    def make_opt(self, stage):
        groups = []
        conn = [p for p in self.conn.parameters() if p.requires_grad]
        groups.append({"params": [p for p in conn if p.ndim >= 2], "lr": LR[stage]["conn"], "weight_decay": 0.01})
        groups.append({"params": [p for p in conn if p.ndim < 2], "lr": LR[stage]["conn"], "weight_decay": 0.0})
        if stage >= 2:
            lora = [p for p in self.llm.parameters() if p.requires_grad]
            groups.append({"params": lora, "lr": LR[stage]["lora"], "weight_decay": 0.01})
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, vlm.lr_lambda(self.stages[stage]))
        return opt, sched

    def state(self):
        sd = {f"conn.{k}": v.detach().cpu() for k, v in self.conn.state_dict().items()}
        sd.update({f"llm.{k}": v for k, v in trainable_state_dict(self.llm).items()})
        return sd

    def load_state(self, sd):
        self.conn.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("conn.")})
        missing = self.llm.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("llm.")},
                                           strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys

    # ------------------------------------------------------------ evaluation helpers
    def nll(self, exs, image_keys, include_eos):
        image_keys = image_keys if image_keys is not None else [e["image"] for e in exs]
        out = []
        self.llm.eval(), self.conn.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(exs), 16):
                b = vlm.collate(self.P, exs[i:i + 16], self.F, image_keys[i:i + 16], include_eos)
                out += vlm.per_example_nll(self.llm, self.conn, self.P, b).tolist()
        self.llm.train(), self.conn.train()
        return out

    def image_keys(self, exs, cond, mapping=None):
        if cond == "correct":
            return [e["image"] for e in exs]
        if cond == "swap":
            return [e["partner_image"] for e in exs]
        if cond == "no_image":
            return [None] * len(exs)
        return [mapping[(e["pair_id"], e["slot"])] for e in exs]

    def dev_eval(self, upd):
        t0 = time.time()
        row = {"update": upd, "stage": self.stage_of(upd),
               "caption_loss": float(np.mean(self.nll(self.cap_dev, None, True)))}
        for suite, exs in self.dev.items():
            nl = {c: self.nll(exs, self.image_keys(exs, c), False) for c in ("correct", "swap", "no_image")}
            for c, v in nl.items():
                row[f"{suite}_nll_{c}"] = float(np.mean(v))
            row[f"{suite}_swap_penalty"] = float(np.mean(np.array(nl["swap"]) - np.array(nl["correct"])))
            for t in sorted({e["answer_type"] for e in exs}):
                idx = [i for i, e in enumerate(exs) if e["answer_type"] == t]
                row[f"{suite}_{t}_swap_penalty"] = float(np.mean([nl["swap"][i] - nl["correct"][i] for i in idx]))
        row["eval_seconds"] = round(time.time() - t0, 1)
        self.dev_logger.log(**row)
        self.dev_rows.append(row)
        print("DEV:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    def stage_of(self, upd):
        for s, (a, b) in self.B.items():
            if a <= upd < b:
                return s
        return 3

    # ------------------------------------------------------------ training
    def train(self):
        self.build()
        self.dev_logger = JsonlLogger(self.res / "dev_log.jsonl")
        self.train_logger = JsonlLogger(self.res / "train_log.jsonl")
        sampler = vlm.Sampler(SEED, self.cap_tr, self.pools)
        u, extra, ck = 0, None, find_latest("run_smoke" if self.smoke else "run_main", self.run_dir)
        if ck is not None:
            st = load_checkpoint(ck)
            u, extra = st["step"], st["extra"]
            if u >= self.B[1][1]:
                self.enable_lora(2)
            self.load_state(st["model"])
            sampler.rng.setstate(extra["sampler_rng"])
            set_rng_state(st["rng"])
            print(f"RESUMED at update {u}", flush=True)
        else:
            self.calibrate()
        self.rows = extra["rows"] if extra else []
        self.dev_rows = extra["dev_rows"] if extra else []
        self.stage_info = extra["stage_info"] if extra else {}
        self.mix = extra["mix"] if extra else {}
        for lg, rows in ((self.train_logger, self.rows), (self.dev_logger, self.dev_rows)):
            lg.path.write_text("")
            for r in rows:
                lg.log(**r)
        save_json(self.checks, self.res / "checks.json")
        if u == 0:
            self.dev_eval(0)

        for s in (1, 2, 3):
            a, b = self.B[s]
            if u >= b:
                continue
            if s >= 2 and not self.has_lora:
                n_mod = self.enable_lora(s)
                print(f"stage {s}: LoRA on {n_mod} modules", flush=True)
            opt, sched = self.make_opt(s)
            scaler = torch.amp.GradScaler("cuda")
            if ck is not None and a < u < b and extra.get("stage") == s:
                opt.load_state_dict(st["optimizer"])
                sched.load_state_dict(st["scheduler"])
                scaler.load_state_dict(st["scaler"])
            info = self.stage_info.get(str(s), {"train_s": 0.0, "peak_gb": 0.0})
            info["trainable_params"] = count_params(self.conn) + (count_params(self.llm) if s >= 2 else 0)
            info["trainable_connector"] = count_params(self.conn)
            info["trainable_lora"] = count_params(self.llm) if s >= 2 else 0
            torch.cuda.reset_peak_memory_stats()
            self.llm.train(), self.conn.train()
            for uu in range(u, b):
                t0 = time.time()
                if s == 1:
                    exs, cnt = sampler.stage1(uu - a, EFF)
                else:
                    exs, cnt = sampler.draw(s, EFF)
                for k, v in cnt.items():
                    self.mix[f"stage{s}_{k}"] = self.mix.get(f"stage{s}_{k}", 0) + v
                total = 0.0
                for m in range(ACCUM):
                    b_ = vlm.collate(self.P, exs[m * MICRO:(m + 1) * MICRO], self.F)
                    with torch.autocast("cuda", dtype=torch.float16):
                        nll = vlm.per_example_nll(self.llm, self.conn, self.P, b_)
                    scaler.scale(nll.sum() / EFF).backward()
                    total += float(nll.detach().sum())
                scaler.unscale_(opt)
                conn_gn = float(torch.norm(torch.stack([p.grad.norm() for p in self.conn.parameters()
                                                        if p.grad is not None])))
                if uu == 0:
                    # Flow check: some connector gradient entry is non-zero. The first step
                    # often overflows in FP16 (inf/NaN norm, update skipped by the scaler);
                    # that still proves gradient reached the connector.
                    flows = any(p.grad is not None and bool((p.grad != 0).any())
                                for p in self.conn.parameters())
                    assert flows, "connector received no gradient"
                    self.checks["first_update_connector_grad_norm"] = conn_gn
                    self.checks["connector_gradient_flows"] = flows
                    save_json(self.checks, self.res / "checks.json")
                gn = float(torch.nn.utils.clip_grad_norm_(self.trainable(), 1.0))
                lrs = [g["lr"] for g in opt.param_groups]
                sc = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                info["train_s"] += time.time() - t0
                upd = uu + 1
                row = {"update": upd, "stage": s, "loss": total / EFF, "grad_norm": gn,
                       "connector_grad_norm": conn_gn, "lr_conn": lrs[0],
                       "lr_lora": lrs[2] if len(lrs) > 2 else 0.0, "loss_scale": sc,
                       "skipped": scaler.get_scale() < sc, **{f"n_{k}": v for k, v in cnt.items()}}
                self.rows.append(row)
                self.train_logger.log(**row)
                if upd % 25 == 0 or self.smoke:
                    print(f"[stage {s}] upd {upd}/{self.total} loss {row['loss']:.4f} gn {gn:.2f} "
                          f"{info['train_s'] / max(1, upd - a):.2f}s/upd peak {gpu_peak_gb():.1f}GB", flush=True)
                if upd % DEV_EVERY == 0 or upd in (self.B[1][1], self.B[2][1], self.B[3][1]):
                    self.dev_eval(upd)
                info["peak_gb"] = max(info["peak_gb"], gpu_peak_gb())
                self.stage_info[str(s)] = info
                if upd % (4 if self.smoke else CKPT_EVERY) == 0 or upd == b:
                    extra = {"sampler_rng": sampler.rng.getstate(), "rows": self.rows,
                             "dev_rows": self.dev_rows, "stage_info": self.stage_info,
                             "mix": self.mix, "stage": s}
                    save_checkpoint(self.run_dir, upd, self.state(), opt, sched, scaler,
                                    data_pos=upd, extra=extra, keep_last=1)
            torch.save(self.state(), self.run_dir / f"stage{s}_weights.pt")
            info["weights_mb"] = round((self.run_dir / f"stage{s}_weights.pt").stat().st_size / 1024**2, 2)
            self.stage_info[str(s)] = info
            u = b
        save_json({"stages": self.stage_info, "mixture_counts": self.mix,
                   "updates": self.stages, "lr": LR, "micro": MICRO, "accum": ACCUM},
                  self.res / "stages.json")

    # ------------------------------------------------------------ final evaluation
    def mismatch_mapping(self, exs):
        pairs = sorted({e["pair_id"] for e in exs}, key=lambda p: [e["pair_id"] for e in exs].index(p))
        imgs = {(e["pair_id"], e["slot"]): e["image"] for e in exs}
        r = random.Random(SEED)
        while True:
            perm = pairs[:]
            r.shuffle(perm)
            if all(a != b for a, b in zip(pairs, perm)):
                break
        mapping = {}
        for p, q in zip(pairs, perm):
            for k in (0, 1):
                mapping[(p, k)] = imgs[(q, k)]
                assert mapping[(p, k)] not in (imgs[(p, 0)], imgs[(p, 1)])
        return mapping

    def final_eval(self):
        if not hasattr(self, "llm"):
            self.build()
            self.enable_lora(2)
            self.load_state(torch.load(self.run_dir / "stage3_weights.pt"))
        self.llm.eval(), self.conn.eval()
        self.llm.config.use_cache = True
        out_dir = self.res / "final"
        out_dir.mkdir(exist_ok=True)
        summary, adv = {}, {}
        for suite, exs in self.test.items():
            mapping = self.mismatch_mapping(exs)
            save_json({f"{p}|{k}": v for (p, k), v in mapping.items()},
                      self.res / f"mismatch_mapping_{suite}.json")
            per = {}
            for cond in CONDS:
                keys = self.image_keys(exs, cond, mapping)
                preds, t_gen = [], 0.0
                torch.cuda.reset_peak_memory_stats()
                for i in range(0, len(exs), 8):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.autocast("cuda", dtype=torch.float16):
                        preds += vlm.generate(self.llm, self.conn, self.P, exs[i:i + 8], self.F, keys[i:i + 8])
                    torch.cuda.synchronize()
                    t_gen += time.time() - t0
                nll = self.nll(exs, keys, False)
                rows = []
                for e, p, n, k in zip(exs, preds, nll, keys):
                    np_, nt = normalize(p), normalize(e["target"])
                    acc = vqa_soft_accuracy(p, e["answers"]) if e["answers"] else float(np_ == nt)
                    rows.append({"pair_id": e["pair_id"], "slot": e["slot"], "type": e["answer_type"],
                                 "question": e["question"], "image_used": k, "reference": e["target"],
                                 "partner_reference": e["partner_target"], "answers": e["answers"],
                                 "prediction": p, "accuracy": acc, "canonical_match": float(np_ == nt),
                                 "follows_partner": float(np_ == normalize(e["partner_target"])),
                                 "empty": float(p.strip() == ""), "nll": n})
                with open(out_dir / f"{suite}_{cond}.jsonl", "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                per[cond] = rows
                pairs = {}
                for r in rows:
                    pairs.setdefault(r["pair_id"], []).append(r)
                s = {"n": len(rows), "accuracy": float(np.mean([r["accuracy"] for r in rows])),
                     "pair_accuracy": float(np.mean([all(x["canonical_match"] for x in v) for v in pairs.values()])),
                     "identical_answer_rate": float(np.mean([normalize(v[0]["prediction"]) == normalize(v[1]["prediction"])
                                                             for v in pairs.values()])),
                     "empty_rate": float(np.mean([r["empty"] for r in rows])),
                     "follows_partner_rate": float(np.mean([r["follows_partner"] for r in rows])),
                     "mean_nll": float(np.mean(nll)),
                     "gen_seconds_per_answer": t_gen / len(rows),
                     "peak_gpu_gb": gpu_peak_gb()}
                for t in sorted({r["type"] for r in rows}):
                    s[f"accuracy_{t}"] = float(np.mean([r["accuracy"] for r in rows if r["type"] == t]))
                summary[f"{suite}/{cond}"] = s
                print(f"FINAL {suite}/{cond}: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                              for k, v in s.items()}), flush=True)
            adv[suite] = self.advantages(per)
        save_json(summary, self.res / "final_summary.json")
        save_json(adv, self.res / "final_advantages.json")
        print("ADVANTAGES:", json.dumps(adv, indent=1), flush=True)

    def advantages(self, per, n_boot=4000):
        """Correct-image advantage over each control, bootstrapping complete pairs."""
        rng = np.random.default_rng(SEED)
        pid = sorted({r["pair_id"] for r in per["correct"]})
        def by_pair(rows, f):
            d = {}
            for r in rows:
                d.setdefault(r["pair_id"], []).append(f(r))
            return np.array([np.mean(d[p]) for p in pid])
        idx = rng.integers(0, len(pid), size=(n_boot, len(pid)))
        out = {}
        for ctrl in ("swap", "mismatched", "no_image"):
            for name, f, sign in (("accuracy", lambda r: r["accuracy"], 1), ("nll", lambda r: r["nll"], -1)):
                d = sign * (by_pair(per["correct"], f) - by_pair(per[ctrl], f))
                boots = d[idx].mean(1)
                out[f"{name}_vs_{ctrl}"] = {"mean": float(d.mean()),
                                            "ci95": [float(np.percentile(boots, 2.5)),
                                                     float(np.percentile(boots, 97.5))]}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    run = Run(args.smoke)
    if not args.eval_only:
        run.train()
    run.final_eval()


if __name__ == "__main__":
    main()