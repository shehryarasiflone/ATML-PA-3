"""SigLIP-Qwen VLM: data loading, feature caching, connector, prompt building, forward/generate."""
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.common.utils import SPLITS_DIR, input_dirs, work_dir
from src.task1.lora import LoRALinear

LLM = "Qwen/Qwen3-0.6B"
VISION = "google/siglip-base-patch16-224"
N_VIS, D_VIS, D_LLM = 196, 768, 1024
IMG_PAD = "<|image_pad|>"
CAPTION_INSTR = "Write a short caption for this image."
QA_SUFFIX = "\nAnswer the question using a single word or phrase."
CAP_Q, CAP_CAPTION, CAP_ANSWER = 80, 64, 16
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


# ------------------------------------------------------------------ data files
def splits_dir() -> Path:
    """splits/task2 in the repo, else copied from an attached t2-data notebook output."""
    d = SPLITS_DIR / "task2"
    if (d / "vqa_train.json").exists():
        return d
    for root in input_dirs():
        for hit in root.rglob("vqa_train.json"):
            if hit.parent.name == "task2":
                shutil.copytree(hit.parent, d, dirs_exist_ok=True)
                return d
    raise FileNotFoundError("splits/task2 not found: attach the t2-data notebook output")


def images_dir() -> Path:
    local = work_dir() / "task2" / "images"
    if (local / "val2014").is_dir():
        return local
    for root in input_dirs():
        for hit in root.rglob("val2014"):
            if hit.is_dir() and hit.parent.name == "images":
                return hit.parent
    raise FileNotFoundError("COCO images not found: attach the t2-data notebook output")


def load_split(name):
    return json.load(open(splits_dir() / f"{name}.json"))


def key(split, image_id):
    return f"{split}/{image_id}"


def question_examples(pairs, split):
    """Flatten pairs into examples that know their partner (same question, other image)."""
    out = []
    for p in pairs:
        pid = p.get("pair_id") or f"{p['kind']}:{p['question']}:{p['examples'][0]['image_id']}"
        ex = p["examples"]
        for k in (0, 1):
            e, o = ex[k], ex[1 - k]
            out.append({
                "type": p["kind"], "answer_type": p.get("answer_type", p["kind"]),
                "pair_id": pid, "slot": k, "question": p["question"],
                "image": key(split, e["image_id"]), "target": e["answer"],
                "answers": e.get("answers"), "partner_image": key(split, o["image_id"]),
                "partner_target": o["answer"]})
    return out


def caption_examples(rows, split):
    return [{"type": "caption", "image": key(split, r["image_id"]), "target": r["caption"],
             "caption_id": r["caption_id"]} for r in rows]


# ------------------------------------------------------------------ features
@torch.no_grad()
def cache_features(image_keys, out_path, batch=64, log=print):
    """SigLIP last_hidden_state = 196 patch features (14x14 row-major) after the vision
    tower's post_layernorm, applied once by the model. Deterministic preprocessing."""
    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel

    proc = SiglipImageProcessor.from_pretrained(VISION)
    enc = SiglipVisionModel.from_pretrained(VISION, dtype=torch.float16).cuda().eval()
    root = images_dir()
    keys = sorted(image_keys)
    feats = torch.empty(len(keys), N_VIS, D_VIS, dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(0, len(keys), batch):
        ims = [Image.open(root / k.split("/")[0] / f"{k.split('/')[1]}.jpg").convert("RGB")
               for k in keys[i:i + batch]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        h = enc(pixel_values=px).last_hidden_state
        assert h.shape[1:] == (N_VIS, D_VIS), h.shape
        feats[i:i + len(ims)] = h.cpu()
        if i % (batch * 20) == 0:
            log(f"  features {i + len(ims)}/{len(keys)}")
    stats = {"images": len(keys), "seconds": round(time.time() - t0, 1),
             "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
             "file_gb": round(feats.numel() * 2 / 1024**3, 2),
             "processor": {"size": proc.size, "mean": proc.image_mean, "std": proc.image_std,
                           "resample": int(proc.resample)}}
    torch.save({"keys": keys, "feats": feats}, out_path)
    del enc
    torch.cuda.empty_cache()
    return stats


class FeatureStore:
    def __init__(self, path):
        d = torch.load(path, weights_only=False)
        self.index = {k: i for i, k in enumerate(d["keys"])}
        self.feats = d["feats"]

    def get(self, keys):
        return self.feats[[self.index[k] for k in keys]]


# ------------------------------------------------------------------ model
class Connector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_VIS, D_LLM)
        self.fc2 = nn.Linear(D_LLM, D_LLM)
        self.register_buffer("scale", torch.tensor(1.0))     # fixed calibration scale

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x))) * self.scale


def add_lora(llm, r=16, alpha=32, dropout=0.05):
    n = 0
    for layer in llm.model.layers:
        for name in LORA_TARGETS:
            setattr(layer.self_attn, name,
                    LoRALinear(getattr(layer.self_attn, name), r, alpha, dropout))
            n += 1
    return n


class Prompter:
    def __init__(self, tok):
        self.tok = tok
        self.pad_id = tok.convert_tokens_to_ids(IMG_PAD)
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.cache = {}

    def prompt_ids(self, ex, with_image=True):
        if ex["type"] == "caption":
            text = CAPTION_INSTR
        else:
            q = self.tok(ex["question"], add_special_tokens=False)["input_ids"][:CAP_Q]
            text = self.tok.decode(q) + QA_SUFFIX
        ck = (text, with_image)
        if ck not in self.cache:
            content = (IMG_PAD * N_VIS + "\n" if with_image else "") + text
            s = self.tok.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
            ids = self.tok(s, add_special_tokens=False)["input_ids"]
            assert sum(i == self.pad_id for i in ids) == (N_VIS if with_image else 0)
            self.cache[ck] = ids
        return self.cache[ck]

    def target_ids(self, ex, target=None):
        cap = CAP_CAPTION if ex["type"] == "caption" else CAP_ANSWER
        t = ex["target"] if target is None else target
        return self.tok(t, add_special_tokens=False)["input_ids"][:cap] + [self.end_id]


def collate(prompter, exs, feats, image_keys=None, include_eos=True, pad_left=False,
            with_target=True):
    """Builds ids/labels/masks. image_keys[i] = feature key or None (no image)."""
    image_keys = image_keys if image_keys is not None else [e["image"] for e in exs]
    seqs, labs = [], []
    for e, ik in zip(exs, image_keys):
        p = prompter.prompt_ids(e, with_image=ik is not None)
        if with_target:
            t = prompter.target_ids(e)
            lab = [-100] * len(p) + (t if include_eos else t[:-1] + [-100])
            seqs.append(p + t)
        else:
            lab = [-100] * len(p)
            seqs.append(p)
        labs.append(lab)
    T = max(len(s) for s in seqs)
    pad = prompter.tok.pad_token_id
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    lab = torch.full((len(seqs), T), -100, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        sl = slice(T - len(s), T) if pad_left else slice(0, len(s))
        ids[i, sl] = torch.tensor(s)
        lab[i, sl] = torch.tensor(l)
        att[i, sl] = 1
    keys = [k for k in image_keys if k is not None]
    vis = feats.get(keys) if keys else None
    return {"ids": ids.cuda(), "labels": lab.cuda(), "att": att.cuda(),
            "vis": vis.cuda() if vis is not None else None}


def embed(llm, connector, prompter, batch):
    emb = llm.get_input_embeddings()(batch["ids"])
    if batch["vis"] is not None:
        v = connector(batch["vis"].float())
        mask = batch["ids"] == prompter.pad_id
        emb = emb.clone()
        emb[mask] = v.reshape(-1, D_LLM).to(emb.dtype)
    return emb


def _ce(h, w, y):
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="none")


def per_example_nll(llm, connector, prompter, batch):
    """Mean supervised-token NLL per example (labels shifted inside)."""
    emb = embed(llm, connector, prompter, batch)
    pos = (batch["att"].cumsum(1) - 1).clamp(min=0)
    h = llm.model(inputs_embeds=emb, attention_mask=batch["att"], position_ids=pos).last_hidden_state
    y = batch["labels"][:, 1:]
    keep = y != -100
    hs = h[:, :-1][keep]
    tok_nll = checkpoint(_ce, hs, llm.lm_head.weight, y[keep], use_reentrant=False)
    row = keep.nonzero()[:, 0]
    n_tok = keep.sum(1).clamp(min=1)
    sums = torch.zeros(keep.shape[0], device=h.device, dtype=torch.float32).index_add_(0, row, tok_nll)
    return sums / n_tok


@torch.no_grad()
def generate(llm, connector, prompter, exs, feats, image_keys, max_new=16):
    batch = collate(prompter, exs, feats, image_keys, pad_left=True, with_target=False)
    emb = embed(llm, connector, prompter, batch)
    out = llm.generate(inputs_embeds=emb, attention_mask=batch["att"], max_new_tokens=max_new,
                       do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=prompter.tok.pad_token_id,
                       eos_token_id=[prompter.end_id, prompter.tok.eos_token_id])
    res = []
    for row in out.tolist():
        if prompter.end_id in row:
            row = row[:row.index(prompter.end_id)]
        res.append(prompter.tok.decode(row, skip_special_tokens=True).strip())
    return res


def lr_lambda(total, warm_frac=0.05, floor=0.1):
    warm = max(1, math.ceil(warm_frac * total))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


class Sampler:
    """Seeded mixture sampler. Stage 1: two shuffled passes over captions.
    Stages 2-3: each draw = one question pair (2 examples) or two caption examples."""
    MIX = {2: [("presence", .50), ("counting", .25), ("caption", .25)],
           3: [("vqa", .70), ("presence", .10), ("counting", .10), ("caption", .10)]}

    def __init__(self, seed, captions, pools):
        self.captions, self.pools = captions, pools
        r = random.Random(seed)
        order = list(range(len(captions)))
        r.shuffle(order)
        o2 = list(range(len(captions)))
        r.shuffle(o2)
        self.stage1_order = order + o2
        self.rng = random.Random(seed + 1)

    def stage1(self, update, n):
        # Full run: 2,250 x 16 = 36,000 = exactly two passes. The modulo only matters for the
        # smoke test, whose small caption subset is shorter than its stage 1.
        o = self.stage1_order
        idx = [o[j % len(o)] for j in range(update * n, (update + 1) * n)]
        return [self.captions[i] for i in idx], {"caption": len(idx)}

    def draw(self, stage, n_examples):
        exs, counts = [], {}
        names, probs = zip(*self.MIX[stage])
        while len(exs) < n_examples:
            src = self.rng.choices(names, probs)[0]
            counts[src] = counts.get(src, 0) + 1
            if src == "caption":
                exs += [self.captions[self.rng.randrange(len(self.captions))] for _ in range(2)]
            else:
                pool = self.pools[src]
                pid = self.rng.randrange(len(pool) // 2)
                exs += pool[2 * pid:2 * pid + 2]           # both members of one pair
        return exs, counts