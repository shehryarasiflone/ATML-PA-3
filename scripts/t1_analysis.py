import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

# No torch needed: paths are set directly so this runs on any CPU machine.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
T1 = RESULTS_DIR / "task1"
EV = T1 / "eval"
OUT = T1 / "analysis"
CONDS = ["baseline", "full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]
RUNS = ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]
LABEL = {"baseline": "Baseline", "full_ft_mid": "Full FT (mid)", "full_ft": "Full FT",
         "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16",
         "oplora16": "OPLoRA-16"}
# Fixed categorical order (validated palette); colour follows the condition everywhere.
COLOR = {"full_ft": "#2a78d6", "l2sp": "#eb6834", "lora4": "#1baf7a", "lora16": "#eda100",
         "oplora16": "#e87ba4", "full_ft_mid": "#008300", "baseline": "#8a8985"}
DASH = {"full_ft": "-", "l2sp": "--", "lora4": "-.", "lora16": ":", "oplora16": (0, (5, 1, 1, 1))}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BENCH = [("gsm8k", "GSM8K", "flexible EM"), ("hellaswag", "HellaSwag", "acc_norm"),
         ("boolq", "BoolQ", "acc"), ("arc_easy", "ARC-Easy", "acc_norm")]

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------- loading
def gsm(cond):
    return pd.read_json(EV / cond / "gsm8k_samples.jsonl", lines=True).sort_values("doc_id")


def mc(cond, task):
    d = pd.read_json(EV / cond / "mc_samples.jsonl", lines=True)
    return d[d.task == task].sort_values("doc_id")


def norm_num(s):
    return (str(s).replace(",", "").replace("$", "").replace("\\%", "").replace("%", "")
            .rstrip(".").strip())


def same(a, b):
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def final_answer(resp, flexible):
    """The model's own stated answer: last \\boxed{...}, else 'The answer is X', else the
    harness's flexible extraction. Diagnostic only; the primary metric stays flexible EM."""
    m = re.findall(r"\\boxed\{([^{}]*)\}", resp)
    if m:
        n = re.findall(r"-?\d[\d,]*\.?\d*", m[-1])
        if n:
            return norm_num(n[-1])
    m = re.findall(r"The answer is:?\s*\$?(-?[\d,]*\.?\d+)", resp)
    if m:
        return norm_num(m[-1])
    return norm_num(flexible)


def correctness(cond, bench):
    if bench == "gsm8k":
        return gsm(cond).correct_flexible.values.astype(int)
    if bench == "gsm8k_final":
        d = gsm(cond)
        return np.array([same(final_answer(r.response, r.extracted_flexible), norm_num(r.target))
                         for r in d.itertuples()]).astype(int)
    d = mc(cond, bench)
    return (d.acc_norm if bench in ("hellaswag", "arc_easy") else d.acc).values.astype(int)


def mcnemar(x, y):
    win, loss = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    p = binomtest(win, win + loss, 0.5).pvalue if win + loss else 1.0
    return win, loss, p


# ---------------------------------------------------------------- tables
def benchmark_table():
    rows = []
    base = {b: correctness("baseline", b) for b, _, _ in BENCH}
    base["gsm8k_final"] = correctness("baseline", "gsm8k_final")
    for c in CONDS:
        row = {"condition": LABEL[c]}
        for b, name, metric in BENCH + [("gsm8k_final", "GSM8K final-answer", "rescored")]:
            v = correctness(c, b)
            row[f"{name}"] = round(100 * v.mean(), 2)
            if c != "baseline":
                row[f"{name} Δpp"] = round(100 * (v.mean() - base[b].mean()), 2)
                row[f"{name} p"] = mcnemar(base[b], v)[2]
        for task, name in (("hellaswag", "HellaSwag"), ("arc_easy", "ARC-Easy")):
            row[f"{name} raw acc"] = round(100 * mc(c, task).acc.mean(), 2)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "benchmark_table.csv", index=False)

    lines = ["| Condition | GSM8K | HellaSwag (norm / raw) | BoolQ | ARC-Easy (norm / raw) "
             "| GSM8K final-answer* |", "|---|---|---|---|---|---|"]
    for r in rows:
        def cell(name, raw=None):
            s = f"{r[name]:.1f}"
            if f"{name} Δpp" in r:
                s += f" ({r[f'{name} Δpp']:+.1f})"
            if raw:
                s = s.replace(f"{r[name]:.1f}", f"{r[name]:.1f} / {r[raw]:.1f}", 1)
            return s
        lines.append(f"| {r['condition']} | {cell('GSM8K')} | "
                     f"{cell('HellaSwag', 'HellaSwag raw acc')} | {cell('BoolQ')} | "
                     f"{cell('ARC-Easy', 'ARC-Easy raw acc')} | {cell('GSM8K final-answer')} |")
    lines.append("\n*Diagnostic rescoring by the model's stated final answer (see gsm8k_format.csv); "
                 "not the primary metric.")
    (OUT / "benchmark_table.md").write_text("\n".join(lines) + "\n")
    return t


def paired_tests():
    pairs = [("baseline", c) for c in CONDS[1:]] + [
        ("full_ft_mid", "full_ft"), ("full_ft", "l2sp"), ("lora4", "lora16"),
        ("lora16", "oplora16"), ("full_ft", "lora16"), ("l2sp", "lora16")]
    rows = []
    for b in ["gsm8k", "gsm8k_final", "hellaswag", "boolq", "arc_easy"]:
        V = {c: correctness(c, b) for c in CONDS}
        for a, c in pairs:
            w, l, p = mcnemar(V[a], V[c])
            rows.append({"benchmark": b, "comparison": f"{c} vs {a}",
                         "delta_pp": round(100 * (V[c].mean() - V[a].mean()), 2),
                         "improved": w, "worsened": l, "p_mcnemar": p})
    pd.DataFrame(rows).to_csv(OUT / "paired_tests.csv", index=False)


def gsm8k_format():
    rows = []
    for c in CONDS:
        d = gsm(c)
        fin = correctness(c, "gsm8k_final")
        rows.append({
            "condition": LABEL[c],
            "flexible_em": round(100 * d.correct_flexible.mean(), 2),
            "strict_em": round(100 * d.correct_strict.mean(), 2),
            "final_answer_acc": round(100 * fin.mean(), 2),
            "extraction_failures": int(((fin == 1) & (d.correct_flexible.values == 0)).sum()),
            "extracted_$$": int((d.extracted_flexible == "$$").sum()),
            "pct_boxed": round(100 * d.response.str.contains(r"\\boxed").mean(), 1),
            "pct_the_answer_is": round(100 * d.response.str.contains("The answer is").mean(), 1),
            "pct_####": round(100 * d.response.str.contains("####").mean(), 1),
            "truncated": int(d.truncated.sum()),
            "truncated_correct": int(d[d.truncated].correct_flexible.sum()),
            "mean_response_tokens": round(d.n_tokens.mean(), 1),
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "gsm8k_format.csv", index=False)
    return t


def boolq_calibration():
    rows = []
    for c in CONDS:
        d = mc(c, "boolq")
        ll = np.array(d.loglik.tolist())            # choices: [" no", " yes"]
        y = d.target.values
        pred = ll.argmax(1)
        margin = ll[:, 1] - ll[:, 0]
        rows.append({"condition": LABEL[c], "acc": round(100 * (pred == y).mean(), 2),
                     "pred_yes_rate": round(100 * pred.mean(), 1),
                     "gold_yes_rate": round(100 * y.mean(), 1),
                     "recall_yes": round(100 * (pred[y == 1] == 1).mean(), 1),
                     "recall_no": round(100 * (pred[y == 0] == 0).mean(), 1),
                     "median_yes_minus_no_logprob": round(float(np.median(margin)), 3),
                     "auc_yes_no_margin": round(roc_auc_score(y, margin), 4)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "boolq_calibration.csv", index=False)
    return t


def resources():
    rows = []
    for r in RUNS:
        s = json.load(open(T1 / r / "summary.json"))
        o = s["overhead"]
        rows.append({
            "method": LABEL[r], "trainable_params": s["trainable_params"],
            "trainable_pct": round(100 * s["trainable_params"] / 596_049_920, 3),
            "train_time_h": s["train_time_h"],
            "sec_per_update": round(3600 * s["train_time_h"] / 1250, 2),
            "peak_gpu_gb": round(s["peak_gpu_gb"], 2),
            "resume_ckpt_mb": round(o["resume_checkpoint_mb"]),
            "inference_export_mb": round(o["export_mb_1250"]),
            "skipped_updates": s["skipped_updates"],
            "method_overhead": "; ".join(f"{k}={v}" for k, v in o.items()
                                         if k in ("reference_storage_mb", "reference_setup_s",
                                                  "basis_preprocessing_s", "basis_storage_mb",
                                                  "max_relative_projection_residual")),
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "resources.csv", index=False)
    return t


def qualitative():
    frames = []
    for c in CONDS:
        d = pd.read_json(EV / c / "qualitative.jsonl", lines=True)
        d["condition"] = c
        frames.append(d)
    q = pd.concat(frames, ignore_index=True)

    def json_valid(s):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
        try:
            json.loads(s)
            return True
        except ValueError:
            return False

    q["json_valid"] = [json_valid(r.response) if r.format == "json" else None
                       for r in q.itertuples()]
    q["math_markers"] = q.response.str.contains(r"####|The answer is|\\boxed")
    q["repetition"] = q.response.map(
        lambda s: (lambda w: len(w) > 40 and len(set(w)) / len(w) < 0.3)(s.split()))
    q.drop(columns=["prompt"]).to_csv(OUT / "qualitative_auto.csv", index=False)

    order = list(dict.fromkeys(q[q.format == "plain"].question))
    subset = [order[0], order[-1]]                 # first and last plain questions
    md = ["# Annotation subset (42 responses)\n",
          "Annotate: format compliance, coherence/relevance, repetition, unnecessary "
          "mathematical framing, inappropriate content, truncation.\n"]
    for qid in subset:
        for fmt in ["plain", "json", "template"]:
            md.append(f"\n## {qid} / {fmt}\n")
            for c in CONDS:
                r = q[(q.question == qid) & (q.format == fmt) & (q.condition == c)].iloc[0]
                flag = " **[TRUNCATED at 512]**" if r.truncated else ""
                md.append(f"\n### {LABEL[c]} ({r.n_new_tokens} tokens){flag}\n\n```\n"
                          f"{r.response}\n```\n")
    (OUT / "annotation_subset.md").write_text("\n".join(md))
    summ = q.groupby("condition").agg(
        truncated=("truncated", "sum"),
        json_valid=("json_valid", lambda x: int(x.dropna().sum())),
        math_markers=("math_markers", "sum"), repetition=("repetition", "sum"),
        mean_tokens=("n_new_tokens", "mean")).reindex(CONDS)
    summ.index = [LABEL[c] for c in summ.index]
    summ.to_csv(OUT / "qualitative_summary.csv")
    return summ


# ---------------------------------------------------------------- figures
def ema(x, a=0.05):
    out, m = [], x[0]
    for v in x:
        m = a * v + (1 - a) * m
        out.append(m)
    return np.array(out)


def fig_learning():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    logs = {r: pd.read_json(T1 / r / "train_log.jsonl", lines=True) for r in RUNS}
    for r in RUNS:
        d = logs[r]
        ax[0].plot(d["update"], ema(d.sft_loss.values), color=COLOR[r], ls=DASH[r], lw=1.4,
                   label=LABEL[r])
    ax[0].set(xlabel="optimizer update", ylabel="SFT loss (EMA 0.05)", ylim=(0.15, 0.45),
              title="(a) Training loss")
    ax[0].legend(fontsize=6.5, loc="upper right")
    for r in ("full_ft", "l2sp"):
        d = logs[r]
        ax[1].plot(d["update"], d.displacement, color=COLOR[r], ls=DASH[r], lw=1.4, label=LABEL[r])
        last = d.displacement.iloc[-1]
        ax[1].annotate(f"{last:.2f}", (1250, last), xytext=(3, 0), textcoords="offset points",
                       va="center", fontsize=6.5, color=INK)
    ax[1].set(xlabel="optimizer update", ylabel=r"$\|\theta-\theta_0\|_2$",
              title="(b) Displacement from start")
    ax[1].legend(fontsize=6.5, loc="lower right")
    d = logs["l2sp"]
    ax[2].plot(d["update"], ema(d.sft_loss.values), color=COLOR["l2sp"], lw=1.4, label="SFT loss")
    ax[2].plot(d["update"], d.lambda_omega, color=MUTED, ls="--", lw=1.4, label=r"$\lambda\Omega$")
    ax[2].set(xlabel="optimizer update", ylabel="loss", ylim=(0, 0.45),
              title=r"(c) L2-SP: SFT vs $\lambda\Omega$")
    ax[2].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_t1_learning.{ext}")
    plt.close(fig)


def fig_benchmarks(table):
    conds = CONDS[1:]
    panels = BENCH + [("gsm8k_final", "GSM8K final-answer", "rescored")]
    fig, ax = plt.subplots(1, 5, figsize=(7.2, 2.4), sharey=False)
    for k, (b, name, metric) in enumerate(panels):
        vals = [table.loc[table.condition == LABEL[c], f"{name} Δpp"].item() for c in conds]
        x = np.arange(len(conds))
        ax[k].bar(x, vals, width=0.78, color=[COLOR[c] for c in conds],
                  edgecolor="white", linewidth=1.0)
        ax[k].axhline(0, color=MUTED, lw=0.8)
        # exact values are in benchmark_table.md; labels here would collide
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = 0.08 * (hi - lo)
        ax[k].set_ylim(lo - pad, hi + pad)
        ax[k].set_xticks(x, [LABEL[c].replace("Full FT + ", "") for c in conds], rotation=60,
                         ha="right", fontsize=6)
        ax[k].set_title(f"{name}\n({metric})", fontsize=7)
        ax[k].grid(axis="x", visible=False)
    ax[0].set_ylabel("change from baseline (pp)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_t1_benchmarks.{ext}")
    plt.close(fig)


def fig_diagnostics():
    D = T1 / "diagnostics"
    ws = pd.read_csv(D / "weight_space.csv")
    act = pd.read_csv(D / "activation_summary.csv")
    conds = CONDS[1:]
    layers = [0, 13, 27]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    w = 0.13
    for j, c in enumerate(conds):
        s = ws[ws.condition == c].set_index("layer")
        x = np.arange(3) + (j - 2.5) * w
        ax[0].bar(x, s.loc[layers, "rho_k"], w * 0.92, color=COLOR[c],
                  label=LABEL[c] + (" (= 0)" if c == "oplora16" else ""))
        ax[1].bar(x, 100 * s.loc[layers, "eta"], w * 0.92, color=COLOR[c])
    ax[0].axhline(128 / 1024, color=MUTED, ls="--", lw=0.9)
    ax[0].annotate("random direction\n(k/d = 0.125)", (2.45, 0.125), xytext=(0, 3),
                   textcoords="offset points", ha="right", va="bottom", fontsize=5.5, color=MUTED)
    for a, t, yl in ((ax[0], r"(a) Subspace interference $\rho_k$", r"$\rho_{128}$"),
                     (ax[1], r"(b) Relative update size $\eta$", r"$\eta$ (%)")):
        a.set_xticks(range(3), [f"layer {l}" for l in layers])
        a.set(title=t, ylabel=yl)
        a.grid(axis="x", visible=False)
    ax[0].legend(fontsize=5.5, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax[0].set_ylim(0, 0.24)
    piv = act.pivot_table(index=["condition", "set"], columns="layer", values="mean")
    x = np.arange(len(conds))
    m = [piv.loc[(c, "math")].mean() for c in conds]
    n = [piv.loc[(c, "non_math")].mean() for c in conds]
    ax[2].bar(x - 0.2, m, 0.38, color=[COLOR[c] for c in conds], label="math (GSM8K)")
    ax[2].bar(x + 0.2, n, 0.38, color="white", edgecolor=[COLOR[c] for c in conds],
              hatch="////", linewidth=1.0, label="non-math (HellaSwag)")
    ax[2].set_xticks(x, [LABEL[c].replace("Full FT + ", "") for c in conds], rotation=60,
                     ha="right", fontsize=6)
    ax[2].set(title=r"(c) Activation correction $R(z)$", ylabel="mean over layers 0/13/27")
    ax[2].legend(fontsize=6, loc="upper left")
    ax[2].set_ylim(0, 0.078)
    ax[2].grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_t1_diagnostics.{ext}")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    t = benchmark_table()
    paired_tests()
    print("=== GSM8K FORMAT ===\n", gsm8k_format().to_string(index=False))
    print("\n=== BOOLQ CALIBRATION ===\n", boolq_calibration().to_string(index=False))
    print("\n=== RESOURCES ===\n", resources().drop(columns="method_overhead").to_string(index=False))
    print("\n=== QUALITATIVE (automatic flags) ===\n", qualitative().to_string())
    fig_learning()
    fig_benchmarks(t)
    fig_diagnostics()
    print("\n" + (OUT / "benchmark_table.md").read_text())


if __name__ == "__main__":
    main()
