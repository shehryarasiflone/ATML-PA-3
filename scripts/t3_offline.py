import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
 
from src.common.utils import RESULTS_DIR, SEED, SPLITS_DIR, input_dirs, save_json
from src.task3.answers import (NO_ANSWER, group_answers, last_boxed, math_correct, math_equal,
                               mathvista_gold, mathvista_normalize)
 
OUT = RESULTS_DIR / "task3" / "offline"
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
BUDGETS = [1, 2, 4, 8, 16, 32]
N_ORDERS = 10
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]
CHECKS = [4, 8, 16]
 
 
def find(folder, name):
    for root in input_dirs() + [Path("/kaggle/working")]:
        for hit in root.rglob(name):
            if hit.parent.name == folder:
                return hit
    raise FileNotFoundError(f"{folder}/{name}: attach the t3-data notebook output")
 
 
def splits_and_orders(ds, ids, save=True):
    ids = sorted(ids)
    perm = np.random.default_rng(SEED).permutation(len(ids))
    order = [ids[i] for i in perm]
    split = {q: ("dev" if k < 200 else "eval") for k, q in enumerate(order)}
    rng = np.random.default_rng(SEED)
    orders = {q: [rng.permutation(32).tolist() for _ in range(N_ORDERS)] for q in ids}
    if save:
        save_json({"rule": "per question in sorted-ID order, ten numpy default_rng(6304) "
                           "permutations of rollout IDs 0..31 (one generator for the dataset)",
                   "orders": orders}, SPLITS_DIR / f"task3_orders_{ds}.json")
    return split, orders, order
 
 
# ------------------------------------------------------------------ loading
def flat(col):
    """pyarrow list column -> (values float32 array, offsets int64 array)."""
    col = col.combine_chunks()
    return np.asarray(col.flatten(), dtype=np.float32), np.asarray(col.offsets, dtype=np.int64)
 
 
def load(ds, limit):
    f = FOLDERS[ds]
    g = pq.read_table(find(f, "generations.parquet"),
                      columns=["sample_id", "rollout_id", "generated_token_ids",
                               "n_generated_text_tokens", "hit_max_new_tokens", "generated_text",
                               "gold_answer", "extracted_answer"])
    p = pq.read_table(find(f, "probability_metrics.parquet"))
    keys = ["sample_id", "rollout_id"]
    gdf = g.select(keys).to_pandas()
    pdf = p.select(keys).to_pandas()
    assert (gdf.values == pdf.values).all(), "row order differs between generations and probabilities"
    checks = {}
    n = np.asarray(g.column("n_generated_text_tokens"))
    lp, lp_off = flat(p.column("logp"))
    kl, kl_off = flat(p.column("kl_u_p"))
    tid_len = np.diff(np.asarray(g.column("generated_token_ids").combine_chunks().offsets))
    checks["rows"] = len(n)
    checks["token_ids_len_mismatch"] = int((tid_len != n).sum())
    checks["logp_len_mismatch"] = int((np.diff(lp_off) != n).sum())
    checks["kl_len_mismatch"] = int((np.diff(kl_off) != n).sum())
    checks["nonfinite_logp"] = int((~np.isfinite(lp)).sum())
    checks["nonfinite_kl"] = int((~np.isfinite(kl)).sum())
    checks["zero_length_rollouts"] = int((n == 0).sum())
    df = g.select(["sample_id", "rollout_id", "n_generated_text_tokens", "hit_max_new_tokens",
                   "gold_answer", "extracted_answer"]).to_pandas()
    df.columns = ["sample_id", "rollout_id", "length", "capped", "gold", "extracted"]
    texts = g.column("generated_text").to_pylist()
    if ds == "math":
        rep = [last_boxed(t) for t in texts]
        checks["extraction_reproduced"] = int(sum(a == b for a, b in zip(rep, df.extracted)))
    v = v_off = None
    if ds == "mathvista":
        a = pq.read_table(find(f, "attention_metrics.parquet"))
        assert (a.select(keys).to_pandas().values == gdf.values).all()
        img, o1 = flat(a.column("img_attn_total"))
        pr, o2 = flat(a.column("prompt_text_attn_total"))
        ge, o3 = flat(a.column("generated_text_attn_total"))
        cnt, o4 = flat(a.column("n_generated_text_tokens"))
        checks["attention_len_mismatch"] = int(sum((np.diff(o) != n).sum() for o in (o1, o2, o3, o4)))
        exp = np.concatenate([np.arange(k) for k in n]).astype(np.float32)
        checks["attention_count_mismatch"] = int((cnt != exp).sum())
        den = img + pr + ge
        v = np.where(den > 0, img / np.where(den > 0, den, 1), np.nan).astype(np.float32)
        v_off = o1
        checks["zero_attention_denominator_tokens"] = int((den <= 0).sum())
    sums = lambda x, off: np.add.reduceat(np.nan_to_num(x), off[:-1]) if len(x) else np.zeros(len(off) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["L"] = np.where(n > 0, sums(lp, lp_off) / np.maximum(n, 1), np.nan)
        df["K"] = np.where(n > 0, sums(kl, kl_off) / np.maximum(n, 1), np.nan)
        if v is not None:
            valid = np.add.reduceat((~np.isnan(v)).astype(np.float64), v_off[:-1])
            df["V"] = np.where(valid > 0, sums(v, v_off) / np.maximum(valid, 1), np.nan)
    tok = {"kl": (kl, kl_off), "v": (v, v_off) if v is not None else None}
    if limit:
        keep = sorted(df.sample_id.unique())[:limit]
        df = df[df.sample_id.isin(keep)]
    return df, tok, checks
 
 
# ------------------------------------------------------------------ grouping and grading
def _math_question(args):
    answers, gold = args
    key = group_answers(answers, math_equal)
    groups = [key[a] if a is not None else NO_ANSWER for a in answers]
    reps = sorted(set(groups) - {NO_ANSWER})
    corr = {r: math_correct(gold, r) for r in reps}
    corr[NO_ANSWER] = False
    return groups, [corr[gp] for gp in groups]
 
 
def grade(ds, df):
    df = df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)
    if ds == "math":
        jobs = [([a if isinstance(a, str) else None for a in g.extracted], g.gold.iloc[0])
                for _, g in df.groupby("sample_id", sort=True)]
        with Pool(os.cpu_count()) as pool:
            res = pool.map(_math_question, jobs, chunksize=8)
        df["group"] = [x for gr, _ in res for x in gr]
        df["correct"] = [x for _, c in res for x in c]
    else:
        from datasets import load_dataset
        meta = load_dataset("AI4Math/MathVista", split="testmini")
        meta = meta.remove_columns([c for c in meta.column_names if "image" in c])
        M = {str(r["pid"]): r for r in meta}
        df["group"] = [mathvista_normalize(a if isinstance(a, str) else None, M[s])
                       for s, a in zip(df.sample_id, df.extracted)]
        gold = {s: mathvista_gold(M[s]) for s in df.sample_id.unique()}
        df["gold_norm"] = df.sample_id.map(gold)
        df["correct"] = (df.group != NO_ANSWER) & (df.group == df.gold_norm)
    return df
 
 
# ------------------------------------------------------------------ selection rules
class Q:
    """Everything about one question's 32 rollouts, indexed by rollout_id."""
 
    def __init__(self, d, tok, row_index):
        d = d.sort_values("rollout_id")
        self.groups = d.group.tolist()
        self.correct = d.correct.values.astype(bool)
        self.length = d.length.values.astype(float)
        self.L = np.nan_to_num(d.L.values, nan=-np.inf)
        self.K = np.nan_to_num(d.K.values, nan=-np.inf)
        self.V = np.nan_to_num(d.V.values, nan=-np.inf) if "V" in d else None
        idx = row_index[d.index.values]
        kl, off = tok["kl"]
        self.k = [kl[off[i]:off[i + 1]] for i in idx]
        if tok["v"] is not None:
            v, voff = tok["v"]
            self.v = [v[voff[i]:voff[i + 1]] for i in idx]
 
    def vote(self, cand, weights=None):
        """Answer group with the largest (weighted) vote; ties -> earliest supporting rollout."""
        W, first = {}, {}
        for pos, i in enumerate(cand):
            g = self.groups[i]
            W[g] = W.get(g, 0.0) + (1.0 if weights is None else weights[pos])
            first.setdefault(g, pos)
        best = max(W, key=lambda g: (W[g], -first[g]))
        return best, W
 
    def top(self, score, cand):
        s = score[cand]
        return int(cand[int(np.argmax(s))])            # argmax takes the earliest maximum
 
    def rwc(self, score, cand):
        if len(cand) == 1:
            return self.groups[cand[0]], {self.groups[cand[0]]: 1.0}
        r = rankdata(-score[cand], method="average")    # rank 1 = best
        w = (len(cand) - r) / (len(cand) - 1)
        return self.vote(cand, w)
 
    def visconf(self, cand):
        """Calibrate KL over the pooled tokens of the current candidates only."""
        pooled = np.concatenate([self.k[i] for i in cand]) if len(cand) else np.array([])
        Z = np.full(32, -np.inf)
        Zt = np.full(32, -np.inf)
        if len(pooled) == 0:
            return Z, Zt
        q05, q95 = np.percentile(pooled, [5, 95])      # linear interpolation (numpy default)
        den = max(q95 - q05, 1e-8)
        for i in cand:
            if len(self.k[i]) == 0:
                continue
            c = np.clip((self.k[i] - q05) / den, 0, 1)
            Z[i] = (np.nanmean(self.v[i]) if np.isfinite(self.V[i]) else 0.0) * c.mean()
            Zt[i] = np.nanmean(c * self.v[i]) if np.any(~np.isnan(self.v[i])) else -np.inf
        return Z, Zt
 
 
def run_selection(ds, Qs, orders, split):
    rows = []
    for sid, q in Qs.items():
        for o, perm in enumerate(orders[sid]):
            for N in BUDGETS:
                cand = np.array(perm[:N])
                oracle = bool(q.correct[cand].any())
                ngroups = len({q.groups[i] for i in cand})
                cost = float(q.length[cand].sum())
                base = {"sample_id": sid, "split": split[sid], "order": o, "N": N,
                        "oracle": oracle, "n_groups": ngroups}
 
                def add(method, g=None, rollout=None, c=cost):
                    if rollout is not None:
                        g = q.groups[rollout]
                        ok = bool(q.correct[rollout])
                    else:
                        ok = bool(q.correct[[i for i in cand if q.groups[i] == g][0]])
                    rows.append({**base, "method": method, "group": g, "correct": ok, "cost": c,
                                 "rollout": -1 if rollout is None else int(rollout)})
 
                add("FirstAns", rollout=int(cand[0]), c=float(q.length[cand[0]]))
                add("SC", g=q.vote(cand)[0])
                add("Top-L", rollout=q.top(q.L, cand))
                add("Top-K", rollout=q.top(q.K, cand))
                add("RWC-K", g=q.rwc(q.K, cand)[0])
                add("Shortest", rollout=q.top(-q.length, cand))
                if q.V is not None:
                    Z, Zt = q.visconf(cand)
                    add("Top-V", rollout=q.top(q.V, cand))
                    add("RWC-V", g=q.rwc(q.V, cand)[0])
                    add("Top-VisConf", rollout=q.top(Z, cand))
                    add("RWC-VisConf", g=q.rwc(Z, cand)[0])
                    add("Top-Ztoken", rollout=q.top(Zt, cand))
                    add("RWC-Ztoken", g=q.rwc(Zt, cand)[0])
    return pd.DataFrame(rows)
 
 
# ------------------------------------------------------------------ Step 3 signals
def signals(ds, Qs):
    out, per_q = {}, []
    for sid, q in Qs.items():
        allc = np.arange(32)
        S = {"length": q.length, "L": q.L, "K": q.K}
        if q.V is not None:
            Z, Zt = q.visconf(allc)
            S.update({"V": q.V, "VisConf": Z, "Ztoken": Zt})
        y = q.correct.astype(float)
        rec = {"sample_id": sid, "frac_correct": y.mean()}
        for name, s in S.items():
            s = np.where(np.isfinite(s), s, np.nanmin(np.where(np.isfinite(s), s, np.nan)) - 1)
            o = np.argsort(-s, kind="stable")
            rec[f"{name}_top_minus_bottom"] = y[o[:8]].mean() - y[o[-8:]].mean()
            rec[f"{name}_spearman"] = (spearmanr(s, y).statistic if 0 < y.mean() < 1 else np.nan)
        per_q.append(rec)
    P = pd.DataFrame(per_q)
    mixed = P[(P.frac_correct > 0) & (P.frac_correct < 1)]
    for c in P.columns:
        if c.endswith("_top_minus_bottom") or c.endswith("_spearman"):
            out[c] = {"mean_all_questions": float(P[c].mean()), "mean_mixed_questions": float(mixed[c].mean()),
                      "n_mixed": int(len(mixed))}
    out["n_questions"] = len(P)
    P.to_csv(OUT / f"signals_per_question_{ds}.csv", index=False)
    return out
 
 
def confidence_vision(Qs, split, order_ids):
    """Per-rollout Pearson(c_t, v_t) with calibration over all 32 rollouts of the question."""
    per_q, ident = [], []
    for sid, q in Qs.items():
        pooled = np.concatenate(q.k)
        q05, q95 = np.percentile(pooled, [5, 95])
        den = max(q95 - q05, 1e-8)
        cors = []
        for i in range(32):
            if len(q.k[i]) < 3:
                continue
            c = np.clip((q.k[i] - q05) / den, 0, 1)
            m = ~np.isnan(q.v[i])
            if m.sum() < 3 or c[m].std() == 0 or q.v[i][m].std() == 0:
                continue
            cors.append(np.corrcoef(c[m], q.v[i][m])[0, 1])
            if m.all():   # identity Z_token - Z = Cov_t(c, v) (population covariance)
                zt, z = np.mean(c * q.v[i]), c.mean() * q.v[i].mean()
                ident.append(abs((zt - z) - np.cov(c, q.v[i], bias=True)[0, 1]))
        per_q.append(np.mean(cors) if cors else np.nan)
    a = np.array(per_q, dtype=float)
    a = a[~np.isnan(a)]
    summary = {"questions": int(len(a)), "median": float(np.median(a)), "mean": float(a.mean()),
               "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
               "fraction_positive": float((a > 0).mean()),
               "identity_max_abs_error": float(max(ident)) if ident else None}
    # illustrative trajectories: first two dev questions (split order) with mixed correctness;
    # for each, the correct and the incorrect rollout with the highest VisConf score
    traj = []
    picked = 0
    for sid in order_ids:
        if split[sid] != "dev" or sid not in Qs:
            continue
        q = Qs[sid]
        if not (0 < q.correct.mean() < 1):
            continue
        Z, _ = q.visconf(np.arange(32))
        pooled = np.concatenate(q.k)
        q05, q95 = np.percentile(pooled, [5, 95])
        den = max(q95 - q05, 1e-8)
        for label, mask in (("correct", q.correct), ("incorrect", ~q.correct)):
            i = int(np.argmax(np.where(mask, Z, -np.inf)))
            c = np.clip((q.k[i] - q05) / den, 0, 1)
            for t in range(len(c)):
                traj.append({"sample_id": sid, "rollout_id": i, "label": label, "t": t,
                             "c": float(c[t]), "v": float(q.v[i][t]), "Z": float(Z[i])})
        picked += 1
        if picked == 2:
            break
    pd.DataFrame(traj).to_csv(OUT / "trajectories_mathvista.csv", index=False)
    return summary
 
 
# ------------------------------------------------------------------ Step 5 adaptive
def adaptive(ds, Qs, orders, split):
    """Precompute, per (question, order, n), what each controller sees; then sweep the grid."""
    rec = []
    for sid, q in Qs.items():
        for o, perm in enumerate(orders[sid]):
            for n in CHECKS + [32]:
                cand = np.array(perm[:n])
                sc, W = q.vote(cand)
                counts = sorted(W.values(), reverse=True)
                unique = len(counts) == 1 or counts[0] > counts[1]
                if q.V is not None:
                    score, _ = q.visconf(cand)
                else:
                    score = q.K
                rw, RW = q.rwc(score, cand)
                rv = sorted(RW.values(), reverse=True)
                margin = (rv[0] - (rv[1] if len(rv) > 1 else 0.0)) / max(sum(rv), 1e-12)
                rec.append({"sample_id": sid, "split": split[sid], "order": o, "n": n,
                            "sc_correct": bool(q.correct[[i for i in cand if q.groups[i] == sc][0]]),
                            "lead_frac": counts[0] / n, "lead_ok": unique and sc != NO_ANSWER,
                            "rwc_agrees": rw == sc, "margin": margin,
                            "cost": float(q.length[cand].sum()),
                            "pool_frac_correct": float(q.correct.mean())})
    R = pd.DataFrame(rec)
 
    def simulate(df, tau, gamma, quality):
        piv = {n: df[df.n == n].set_index(["sample_id", "order"]) for n in CHECKS + [32]}
        stop = pd.Series(32, index=piv[32].index)
        decided = pd.Series(False, index=piv[32].index)
        if tau is not None:
            for n in CHECKS:
                d = piv[n]
                ok = d.lead_ok & (d.lead_frac >= tau - 1e-12)
                if quality:
                    ok &= d.rwc_agrees & (d.margin >= gamma - 1e-12)
                new = ok & ~decided
                stop[new] = n
                decided |= ok
        res = pd.DataFrame({"stop": stop})
        for col in ("sc_correct", "cost"):
            res[col] = [piv[s].at[k, col] for k, s in zip(res.index, res.stop)]
        res["fixed32_correct"] = piv[32].sc_correct
        return res
 
    def summarize(res):
        g = res.groupby(level="sample_id")
        return {"accuracy": float(g.sc_correct.mean().mean()), "mean_cost": float(g.cost.mean().mean()),
                "mean_rollouts": float(g.stop.mean().mean()),
                "stop_distribution": {str(k): float(v) for k, v in res.stop.value_counts(normalize=True).sort_index().items()}}
 
    dev, ev = R[R.split == "dev"], R[R.split == "eval"]
    fixed32_dev = summarize(simulate(dev, None, None, False))["accuracy"]
    out = {"fixed32_dev_accuracy": fixed32_dev, "families": {}}
    for fam, quality in (("agreement", False), ("quality_aware", True)):
        grid = [(None, None)] + [(t, g) for t in TAUS for g in (GAMMAS if quality else [None])]
        rows = []
        for t, g in grid:
            s = summarize(simulate(dev, t, g or 0.0, quality))
            rows.append({"tau": t, "gamma": g, **{k: s[k] for k in ("accuracy", "mean_cost", "mean_rollouts")}})
        G = pd.DataFrame(rows)
        feas = G[G.accuracy >= fixed32_dev - 0.01 - 1e-12].copy()
        feas["tau_s"] = feas.tau.fillna(9)
        feas["gamma_s"] = feas.gamma.fillna(0)
        best = feas.sort_values(["mean_cost", "accuracy", "tau_s", "gamma_s"],
                                ascending=[True, False, True, True]).iloc[0]
        t, g = (None if pd.isna(best.tau) else float(best.tau)), (None if pd.isna(best.gamma) else float(best.gamma))
        res = simulate(ev, t, g or 0.0, quality)
        res_df = res.reset_index()
        res_df["policy"] = fam
        early_wrong = res_df[(res_df.stop < 32) & (~res_df.sc_correct.astype(bool))]
        out["families"][fam] = {
            "dev_grid": G.to_dict("records"), "selected": {"tau": t, "gamma": g},
            "eval": summarize(res),
            "eval_incorrect_early_stops": int(len(early_wrong)),
            "eval_incorrect_early_stops_fixed32_correct": int(early_wrong.fixed32_correct.astype(bool).sum()),
            "eval_cases": int(len(res_df))}
        res_df.merge(ev[ev.n == 32][["sample_id", "order", "pool_frac_correct"]],
                     on=["sample_id", "order"]).to_csv(OUT / f"adaptive_cases_{ds}_{fam}.csv", index=False)
    fx = ev.groupby(["n", "sample_id"])[["sc_correct", "cost"]].mean().groupby("n").mean()
    out["fixed_eval"] = {str(n): {"accuracy": float(r.sc_correct), "mean_cost": float(r.cost)}
                         for n, r in fx.iterrows()}
    return out
 
 
# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import importlib.metadata as im
    checks = {"math_verify_version": im.version("math-verify")}
    all_splits = {}
    for ds in args.datasets:
        t0 = time.time()
        df, tok, ck = load(ds, args.limit)
        checks[ds] = ck
        print(f"[{ds}] loaded {len(df)} rollouts; checks {ck}", flush=True)
        df = df.reset_index().rename(columns={"index": "row"})
        ids = sorted(df.sample_id.unique())
        split, orders, order_ids = splits_and_orders(ds, ids, save=not args.limit)
        all_splits[ds] = {"dev": [q for q in order_ids if split[q] == "dev"],
                          "eval": [q for q in order_ids if split[q] == "eval"]}
        df = grade(ds, df)
        df["split"] = df.sample_id.map(split)
        print(f"[{ds}] graded in {time.time() - t0:.0f}s: rollout accuracy {df.correct.mean():.4f}, "
              f"NO_ANSWER {(df.group == NO_ANSWER).mean():.4f}", flush=True)
        df.drop(columns=[c for c in ("gold_norm",) if c in df]).to_parquet(OUT / f"rollouts_{ds}.parquet", index=False)
        df.index = df.row.values
        Qs = {sid: Q(d, tok, np.arange(len(tok["kl"][1]) - 1)) for sid, d in df.groupby("sample_id")}
        t1 = time.time()
        sel = run_selection(ds, Qs, orders, split)
        sel.to_parquet(OUT / f"selections_{ds}.parquet", index=False)
        print(f"[{ds}] selection done in {time.time() - t1:.0f}s", flush=True)
        acc = (sel[sel.split == "eval"].groupby(["method", "N", "sample_id"]).correct.mean()
               .groupby(["method", "N"]).mean().unstack("N").round(4))
        print(f"[{ds}] EVAL ACCURACY (avg over orders, then questions)\n{acc.to_string()}", flush=True)
        sig = signals(ds, Qs)
        if ds == "mathvista":
            sig["confidence_vision_correlation"] = confidence_vision(Qs, split, order_ids)
        save_json(sig, OUT / f"signals_{ds}.json")
        ad = adaptive(ds, Qs, orders, split)
        save_json(ad, OUT / f"adaptive_{ds}.json")
        print(f"[{ds}] ADAPTIVE selected: " + json.dumps({k: (v['selected'], v['eval']['accuracy'], v['eval']['mean_cost'])
                                                         for k, v in ad['families'].items()}), flush=True)
        print(f"[{ds}] total {time.time() - t0:.0f}s", flush=True)
    save_json(checks, OUT / "checks.json")
    if not args.limit:
        save_json(all_splits, SPLITS_DIR / "task3_offline_splits.json")
    print("DONE", flush=True)
 
 
if __name__ == "__main__":
    main()