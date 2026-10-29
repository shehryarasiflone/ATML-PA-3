"""Step 0: check the Kaggle environment and measure Task 1 speed/memory before real runs.

Run:  python scripts/env_check.py
Writes results/env/env_check.json and results/env/pip_freeze.txt
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.utils import RESULTS_DIR, SEED, gpu_peak_gb, save_json, set_seed

MODEL = "Qwen/Qwen3-0.6B"
out = {}


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# 1. Hardware and library versions --------------------------------------------
section("1. Environment")
out["python"] = sys.version.split()[0]
out["torch"] = torch.__version__
out["cuda"] = torch.version.cuda
out["transformers"] = transformers.__version__
for pkg in ["peft", "accelerate", "datasets", "lm_eval", "math_verify"]:
    try:
        out[pkg] = __import__(pkg).__version__
    except Exception as e:  # noqa: BLE001
        out[pkg] = f"MISSING ({e.__class__.__name__})"
out["gpus"] = [
    {"name": torch.cuda.get_device_name(i),
     "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1)}
    for i in range(torch.cuda.device_count())
]
for k, v in out.items():
    print(f"{k:14s} {v}")

env_dir = RESULTS_DIR / "env"
env_dir.mkdir(parents=True, exist_ok=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
(env_dir / "pip_freeze.txt").write_text(freeze)

assert torch.cuda.is_available(), "No GPU. Set Accelerator = GPU T4 x2 in notebook settings."

# 2. Chat template with thinking disabled ------------------------------------
section("2. Chat template (enable_thinking=False)")
tok = AutoTokenizer.from_pretrained(MODEL)
msgs = [{"role": "user", "content": "What is 2+3?"},
        {"role": "assistant", "content": "2+3=5. The answer is 5."}]
full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
print("FULL CONVERSATION:\n" + repr(full))
print("\nGENERATION PROMPT:\n" + repr(prompt))
out["template_full"] = full
out["template_prompt"] = prompt
out["prompt_is_prefix_of_full"] = full.startswith(prompt)
out["eos_token"] = tok.eos_token
out["pad_token"] = tok.pad_token
print("\nprompt is a prefix of full conversation:", out["prompt_is_prefix_of_full"])

# 3. Quick generation sanity check -------------------------------------------
section("3. Generation (FP16)")
set_seed(SEED)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
out["tie_word_embeddings"] = bool(model.config.tie_word_embeddings)
out["num_layers"] = model.config.num_hidden_layers
out["vocab_size"] = model.config.vocab_size
ids = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    g = model.generate(**ids, max_new_tokens=40, do_sample=False)
reply = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
print("reply:", repr(reply))
out["sample_reply"] = reply
print("tied embeddings:", out["tie_word_embeddings"], "| layers:", out["num_layers"])
del model
torch.cuda.empty_cache()

# 4. Full-FT speed and memory (the most expensive Task 1 setting) -------------
section("4. Full-FT step benchmark: FP32 master weights, FP16 autocast, grad checkpointing")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda()
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
scaler = torch.amp.GradScaler("cuda")
bench = {}
for seq_len in [512, 1024, 2048]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randint(0, 50000, (2, seq_len), device="cuda")   # microbatch 2
        times = []
        for i in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=x).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        bench[seq_len] = {"sec_per_microbatch": round(sum(times[1:]) / 2, 3),
                          "peak_gb": gpu_peak_gb()}
    except torch.cuda.OutOfMemoryError:
        bench[seq_len] = "OOM"
        opt.zero_grad(set_to_none=True)
    print(f"seq_len {seq_len:5d}: {bench[seq_len]}")
out["fullft_benchmark_microbatch2"] = bench

save_json(out, env_dir / "env_check.json")
section("Done. Saved results/env/env_check.json — paste sections 2 and 4 back to Claude.")
