import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen, urlretrieve
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
from src.common.utils import SEED, SPLITS_DIR, save_json, work_dir
from src.task2.vqa_norm import normalize
 
RAW = work_dir() / "task2" / "raw"
IMG = work_dir() / "task2" / "images"
OUT = SPLITS_DIR / "task2"
URLS = {
    "coco_ann": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "vqa_q_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "vqa_q_val": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "vqa_a_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "vqa_a_val": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "vqa_p_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "vqa_p_val": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
}
N_CAP_TRAIN, CAPS_PER_IMG, N_CAP_DEV = 6000, 3, 96
GROUND = {"dev": 24, "test": 64}                       # pairs, half presence / half counting
POOL_PRESENCE, POOL_COUNTING = 6000, 6000              # training pair pools
VQA_N = {"train": {"yes/no": 640, "number": 256, "other": 640},
         "dev": {"yes/no": 12, "number": 8, "other": 12},
         "test": {"yes/no": 36, "number": 24, "other": 36}}
MIN_AREA = 1024                                        # 32 x 32 px
EXCLUDE_COUNT = {"broccoli", "carrot", "banana", "book"}
PLURAL = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep",
          "skis": "pairs of skis", "scissors": "pairs of scissors", "tv": "TVs",
          "hair drier": "hair driers"}
 
 
def fetch(name):
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / f"{name}.zip"
    if not zpath.exists():
        print(f"downloading {name} ...", flush=True)
        urlretrieve(URLS[name], zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW)
 
 
def load(pattern):
    hits = sorted(RAW.rglob(pattern))
    assert hits, f"missing {pattern}"
    return json.load(open(hits[0]))
 
 
def plural(name):
    if name in PLURAL:
        return PLURAL[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"
 
 
def article(name):
    return "an" if name[0] in "aeiou" else "a"
 
 
# ---------------------------------------------------------------- COCO grounding
def instance_index(inst):
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    by_img = defaultdict(lambda: defaultdict(list))
    for a in inst["annotations"]:
        by_img[a["image_id"]][a["category_id"]].append(a)
    return cats, by_img
 
 
def presence_candidates(images, cats, by_img):
    yes, no = defaultdict(list), defaultdict(list)
    for im in images:
        anns = by_img.get(im, {})
        for c in cats:
            lst = anns.get(c, [])
            if not lst:
                no[c].append(im)
            elif any(not a["iscrowd"] and a["area"] >= MIN_AREA for a in lst):
                yes[c].append(im)
    return yes, no
 
 
def count_candidates(images, cats, by_img):
    out = defaultdict(lambda: defaultdict(list))        # cat -> count -> images
    for im in images:
        for c, lst in by_img.get(im, {}).items():
            if cats[c] in EXCLUDE_COUNT:
                continue
            if all(not a["iscrowd"] and a["area"] >= MIN_AREA for a in lst) and 1 <= len(lst) <= 4:
                out[c][len(lst)].append(im)
    return out
 
 
def presence_q(name):
    return f"Does this image contain {article(name)} {name}?"
 
 
def count_q(name):
    return f"How many {plural(name)} can you see?"
 
 
def make_presence_pair(rng, c, cats, yes, no, used=None):
    pool_y = [i for i in yes[c] if used is None or i not in used]
    pool_n = [i for i in no[c] if used is None or i not in used]
    if not pool_y or not pool_n:
        return None
    y, n = rng.choice(pool_y), rng.choice(pool_n)
    q = presence_q(cats[c])
    items = [(y, "yes"), (n, "no")]
    rng.shuffle(items)
    return {"kind": "presence", "category": cats[c], "question": q,
            "examples": [{"image_id": i, "answer": a} for i, a in items]}
 
 
def make_count_pair(rng, c, cats, cnt, used=None):
    by_n = {k: [i for i in v if used is None or i not in used] for k, v in cnt[c].items()}
    ks = sorted(k for k, v in by_n.items() if v)
    if len(ks) < 2:
        return None
    k1, k2 = rng.sample(ks, 2)
    i1, i2 = rng.choice(by_n[k1]), rng.choice(by_n[k2])
    if i1 == i2:
        return None
    return {"kind": "counting", "category": cats[c], "question": count_q(cats[c]),
            "examples": [{"image_id": i1, "answer": str(k1)}, {"image_id": i2, "answer": str(k2)}]}
 
 
def build_pool(rng, n, maker, cat_ids, used=None, max_tries=200000):
    pairs, tries = [], 0
    while len(pairs) < n and tries < max_tries:
        tries += 1
        p = maker(rng, rng.choice(cat_ids))
        if p is None:
            continue
        ims = [e["image_id"] for e in p["examples"]]
        if used is not None:
            if any(i in used for i in ims):
                continue
            used.update(ims)
        pairs.append(p)
    assert len(pairs) == n, f"only {len(pairs)}/{n} pairs"
    return pairs
 
 
# ---------------------------------------------------------------- VQAv2
def vqa_pairs(split, qs, anns, pairs_raw, allowed_images):
    q_by = {q["question_id"]: q for q in qs["questions"]}
    a_by = {a["question_id"]: a for a in anns["annotations"]}
    out = defaultdict(list)
    for qa, qb in pairs_raw:
        if qa not in a_by or qb not in a_by:
            continue
        A, B = a_by[qa], a_by[qb]
        if A["image_id"] not in allowed_images or B["image_id"] not in allowed_images:
            continue
        if A["answer_type"] != B["answer_type"]:
            continue
        if q_by[qa]["question"].strip() != q_by[qb]["question"].strip():
            continue
        ex = []
        for qid, ann in ((qa, A), (qb, B)):
            raw = [x["answer"] for x in ann["answers"]]
            norm = [normalize(x) for x in raw]
            cons, n = Counter(norm).most_common(1)[0]
            if n < 8 or len(raw) != 10:
                break
            target = Counter(r.strip() for r, m in zip(raw, norm) if m == cons).most_common(1)[0][0]
            ex.append({"question_id": qid, "image_id": ann["image_id"], "answer": target,
                       "consensus": cons, "agreement": n, "answers": raw})
        else:
            if ex[0]["consensus"] != ex[1]["consensus"] and ex[0]["image_id"] != ex[1]["image_id"]:
                out[A["answer_type"]].append({"kind": "vqa", "answer_type": A["answer_type"],
                                              "question": q_by[qa]["question"].strip(),
                                              "pair_id": f"{qa}_{qb}", "examples": ex})
    print(f"  {split}: eligible VQA pairs " + str({k: len(v) for k, v in out.items()}))
    return out
 
 
def sample_vqa(rng, cands, quota, used):
    chosen = []
    for t, n in quota.items():
        lst = sorted(cands[t], key=lambda p: p["pair_id"])
        rng.shuffle(lst)
        got = 0
        for p in lst:
            ims = [e["image_id"] for e in p["examples"]]
            if any(i in used for i in ims):
                continue
            used.update(ims)
            chosen.append(p)
            got += 1
            if got == n:
                break
        assert got == n, f"only {got}/{n} {t} pairs"
    return chosen
 
 
# ---------------------------------------------------------------- images
def image_url(split, im):
    return f"http://images.cocodataset.org/{split}/COCO_{split}_{im:012d}.jpg"
 
 
def download_images(ids_by_split):
    jobs = []
    for split, ids in ids_by_split.items():
        d = IMG / split
        d.mkdir(parents=True, exist_ok=True)
        for im in sorted(ids):
            p = d / f"{im}.jpg"
            if not p.exists():
                jobs.append((image_url(split, im), p))
    print(f"downloading {len(jobs)} images ...", flush=True)
 
    def get(job):
        url, p = job
        for _ in range(4):
            try:
                data = urlopen(url, timeout=60).read()
                tmp = p.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.rename(p)
                return True
            except Exception:  # noqa: BLE001
                continue
        return False
 
    with ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(get, jobs))
    failed = [str(j[1]) for j, o in zip(jobs, ok) if not o]
    print(f"images done, {len(failed)} failed", flush=True)
    return failed
 
 
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for k in URLS:
        fetch(k)
    rng = random.Random(SEED)
 
    cap_tr, cap_va = load("captions_train2014.json"), load("captions_val2014.json")
    inst_tr, inst_va = load("instances_train2014.json"), load("instances_val2014.json")
 
    # val2014 halves: dev draws from A, final test from B
    val_ids = sorted(im["id"] for im in inst_va["images"])
    rng.shuffle(val_ids)
    half = len(val_ids) // 2
    dev_half, test_half = set(val_ids[:half]), set(val_ids[half:])
    train_ids = set(im["id"] for im in inst_tr["images"])
 
    # ---- VQAv2 natural pairs
    vtr = vqa_pairs("train", load("v2_OpenEnded_mscoco_train2014_questions.json"),
                    load("v2_mscoco_train2014_annotations.json"),
                    load("v2_mscoco_train2014_complementary_pairs.json"), train_ids)
    qv, av = load("v2_OpenEnded_mscoco_val2014_questions.json"), load("v2_mscoco_val2014_annotations.json")
    pv = load("v2_mscoco_val2014_complementary_pairs.json")
    vdev = vqa_pairs("dev", qv, av, pv, dev_half)
    vtest = vqa_pairs("test", qv, av, pv, test_half)
    used_dev, used_test = set(), set()
    vqa = {"train": sample_vqa(rng, vtr, VQA_N["train"], set()),
           "dev": sample_vqa(rng, vdev, VQA_N["dev"], used_dev),
           "test": sample_vqa(rng, vtest, VQA_N["test"], used_test)}
 
    # ---- captions
    caps = defaultdict(list)
    for a in cap_tr["annotations"]:
        caps[a["image_id"]].append(a)
    cap_imgs = sorted(i for i in caps if len(caps[i]) >= CAPS_PER_IMG)
    cap_imgs = sorted(rng.sample(cap_imgs, N_CAP_TRAIN))
    cap_train = []
    for im in cap_imgs:
        for a in rng.sample(sorted(caps[im], key=lambda a: a["id"]), CAPS_PER_IMG):
            cap_train.append({"image_id": im, "caption_id": a["id"], "caption": a["caption"].strip()})
    vcaps = defaultdict(list)
    for a in cap_va["annotations"]:
        vcaps[a["image_id"]].append(a)
    dev_cap_pool = sorted(i for i in vcaps if i in dev_half and i not in used_dev)
    cap_dev = []
    for im in sorted(rng.sample(dev_cap_pool, N_CAP_DEV)):
        a = rng.choice(sorted(vcaps[im], key=lambda a: a["id"]))
        cap_dev.append({"image_id": im, "caption_id": a["id"], "caption": a["caption"].strip()})
        used_dev.add(im)
 
    # ---- annotation-derived grounding
    cats_tr, by_tr = instance_index(inst_tr)
    cats_va, by_va = instance_index(inst_va)
    pool_imgs = sorted(set(cap_imgs) | {e["image_id"] for p in vqa["train"] for e in p["examples"]})
    yes_tr, no_tr = presence_candidates(pool_imgs, cats_tr, by_tr)
    cnt_tr = count_candidates(pool_imgs, cats_tr, by_tr)
    pcats = sorted(c for c in cats_tr if yes_tr[c] and no_tr[c])
    ccats = sorted(c for c in cnt_tr if sum(1 for v in cnt_tr[c].values() if v) >= 2)
    ground_train = {
        "presence": build_pool(rng, POOL_PRESENCE,
                               lambda r, c: make_presence_pair(r, c, cats_tr, yes_tr, no_tr), pcats),
        "counting": build_pool(rng, POOL_COUNTING,
                               lambda r, c: make_count_pair(r, c, cats_tr, cnt_tr), ccats),
    }
    ground_eval = {}
    for split, half_ids, used in (("dev", dev_half, used_dev), ("test", test_half, used_test)):
        imgs = sorted(half_ids)
        yes, no = presence_candidates(imgs, cats_va, by_va)
        cnt = count_candidates(imgs, cats_va, by_va)
        pc = sorted(c for c in cats_va if yes[c] and no[c])
        cc = sorted(c for c in cnt if sum(1 for v in cnt[c].values() if v) >= 2)
        n = GROUND[split] // 2
        ground_eval[split] = (
            build_pool(rng, n, lambda r, c: make_presence_pair(r, c, cats_va, yes, no, used), pc,
                       used=used)
            + build_pool(rng, n, lambda r, c: make_count_pair(r, c, cats_va, cnt, used), cc,
                         used=used))
 
    # ---- checks and saving
    assert not (used_dev & used_test), "dev and test share images"
    for name, obj in [("captions_train", cap_train), ("captions_dev", cap_dev),
                      ("grounding_train_pools", ground_train),
                      ("grounding_dev", ground_eval["dev"]), ("grounding_test", ground_eval["test"]),
                      ("vqa_train", vqa["train"]), ("vqa_dev", vqa["dev"]), ("vqa_test", vqa["test"])]:
        save_json(obj, OUT / f"{name}.json")
    imgs_train = set(pool_imgs)
    imgs_val = used_dev | used_test
    stats = {
        "seed": SEED, "min_area_px": MIN_AREA, "exclude_counting": sorted(EXCLUDE_COUNT),
        "captions_train": len(cap_train), "captions_dev": len(cap_dev),
        "grounding_pool_pairs": {k: len(v) for k, v in ground_train.items()},
        "grounding_pool_categories": {"presence": len(pcats), "counting": len(ccats)},
        "grounding_dev_pairs": Counter(p["kind"] for p in ground_eval["dev"]),
        "grounding_test_pairs": Counter(p["kind"] for p in ground_eval["test"]),
        "vqa_pairs": {s: Counter(p["answer_type"] for p in v) for s, v in vqa.items()},
        "counting_answer_dist_train_pool": Counter(e["answer"] for p in ground_train["counting"]
                                                   for e in p["examples"]),
        "images": {"train2014": len(imgs_train), "val2014_dev": len(used_dev),
                   "val2014_test": len(used_test)},
    }
    save_json(stats, OUT / "data_stats.json")
    print(json.dumps(stats, indent=1, default=str))
 
    failed = download_images({"train2014": imgs_train, "val2014": imgs_val})
    save_json({"failed": failed}, OUT / "image_download_failures.json")
    print("\nEXAMPLES")
    for name, obj in [("grounding_dev", ground_eval["dev"][:2] + ground_eval["dev"][-2:]),
                      ("vqa_dev", vqa["dev"][:3])]:
        for p in obj:
            print(name, p["question"], [(e["image_id"], e["answer"]) for e in p["examples"]])
 
 
if __name__ == "__main__":
    main()