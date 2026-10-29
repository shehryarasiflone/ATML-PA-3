import re
import unicodedata
import torch
from datasets import load_dataset
from src.common.utils import SPLITS_DIR, save_json, work_dir

MODEL = "Qwen/Qwen3-0.6B"
N_TRAIN = 40_000
MAX_LEN = 2048
CACHE = "task1/train_tokenized.pt"


def normalize(text: str) -> str:
    """NFKC, lowercase, keep only letters/digits, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def format_example(tok, question: str, answer: str):
    """Returns (input_ids, response_start) or (None, reason) if the split is unsafe."""
    user = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": answer}],
                                   tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):
        return None, "prompt_not_prefix"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:
        return None, "token_boundary_mismatch"
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    last_end = len(f_ids) - 1 - f_ids[::-1].index(end_id)
    if last_end < len(p_ids):
        return None, "no_response_tokens"
    # Keep through <|im_end|>; drop the template's trailing newline (not end-of-turn).
    return f_ids[: last_end + 1], len(p_ids)


def build_train_set(tok, log=print):
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_norm = {normalize(q) for q in gsm["question"]}
    log(f"GSM8K test questions: {len(gsm)} ({len(gsm_norm)} unique after normalization)")

    mm = load_dataset("meta-math/MetaMathQA", split="train")
    log(f"MetaMathQA rows: {len(mm)}")

    kept, excluded = [], {}
    def skip(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for idx, ex in enumerate(mm):
        if len(kept) == N_TRAIN:
            last_scanned = idx - 1
            break
        q, a = (ex.get("query") or "").strip(), (ex.get("response") or "").strip()
        if not q or not a:
            skip("empty_question_or_response"); continue
        if normalize(q) in gsm_norm or normalize(ex.get("original_question")) in gsm_norm:
            skip("gsm8k_test_overlap"); continue
        ids, start = format_example(tok, ex["query"], ex["response"])
        if ids is None:
            skip(start); continue
        if len(ids) > MAX_LEN:
            skip("longer_than_2048"); continue
        kept.append({"src_idx": idx, "type": ex.get("type"),
                     "input_ids": ids, "response_start": start})
    else:
        last_scanned = len(mm) - 1
    assert len(kept) == N_TRAIN, f"only {len(kept)} eligible examples"
    return kept, excluded, last_scanned


def length_stats(kept) -> dict:
    lens = torch.tensor([len(k["input_ids"]) for k in kept], dtype=torch.float)
    resp = torch.tensor([len(k["input_ids"]) - k["response_start"] for k in kept],
                        dtype=torch.float)
    q = lambda t, p: round(float(torch.quantile(t, p)), 1)
    return {
        "total_tokens": {"mean": round(float(lens.mean()), 1), "median": q(lens, .5),
                         "p90": q(lens, .9), "p99": q(lens, .99), "max": int(lens.max())},
        "response_tokens": {"mean": round(float(resp.mean()), 1), "median": q(resp, .5),
                            "sum": int(resp.sum())},
        "n_over_512": int((lens > 512).sum()),
        "n_over_1024": int((lens > 1024).sum()),
    }


def load_or_build(tok, log=print):
    """Used by the trainer: reuse the cached tokenization, rebuild if missing."""
    path = work_dir() / CACHE
    if path.exists():
        return torch.load(path, weights_only=False)
    kept, excluded, last = build_train_set(tok, log)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kept, path)
    save_json({"source_indices": [k["src_idx"] for k in kept]},
              SPLITS_DIR / "task1_train_source_indices.json")
    return kept