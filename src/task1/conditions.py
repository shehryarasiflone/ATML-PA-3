from pathlib import Path
 
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
 
from src.common.utils import input_dirs, work_dir
from src.task1.data import MODEL
from src.task1.lora import LoRALinear, inject_adapters
 
# condition -> (kind, training run, export update)
CONDITIONS = {
    "baseline":    ("base", None, None),
    "full_ft_mid": ("full", "full_ft", 625),
    "full_ft":     ("full", "full_ft", 1250),
    "l2sp":        ("full", "l2sp", 1250),
    "lora4":       ("lora", "lora4", 1250),
    "lora16":      ("lora", "lora16", 1250),
    "oplora16":    ("lora", "oplora16", 1250),
}
 
 
def find_export(run: str, update: int) -> Path:
    """Export folder in this session's outputs or in any attached notebook output."""
    name = f"export_update_{update:04d}"
    local = work_dir() / "task1" / run / name
    if local.is_dir():
        return local
    for d in input_dirs():
        for hit in d.rglob(name):
            if hit.is_dir() and hit.parent.name == run:
                return hit
    raise FileNotFoundError(
        f"No {run}/{name}. Attach the training notebook that produced it (Add Input).")
 
 
def effective_factors(entry: dict):
    """(A, B, scale) used for inference: projected factors for OPLoRA, raw otherwise."""
    A = entry.get("A_proj", entry["A"]).float()
    B = entry.get("B_proj", entry["B"]).float()
    return A, B, float(entry["scale"])
 
 
def load_adapters(run: str) -> dict:
    return torch.load(find_export(run, 1250) / "adapters.pt", map_location="cpu",
                      weights_only=False)
 
 
def _finish(model, device):
    model.config.use_cache = True                      # training config saved use_cache=False
    # Same generation defaults (eos ids etc.) for every condition; lm_eval forces greedy.
    model.generation_config = GenerationConfig.from_pretrained(MODEL)
    return model.to(device).eval()
 
 
def load_condition(cond: str, device="cuda"):
    """FP16 model for evaluation. Adapters are merged into a fresh backbone copy."""
    kind, run, upd = CONDITIONS[cond]
    if kind == "base":
        return _finish(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16), device)
    if kind == "full":
        path = find_export(run, upd)
        return _finish(AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16), device)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    with torch.no_grad():
        for name, entry in load_adapters(run).items():
            A, B, s = effective_factors(entry)
            model.get_submodule(name).weight.add_(s * (B @ A))
    return _finish(model.half(), device)
 
 
def load_unmerged(run: str, device="cuda"):
    """FP16 backbone with FP32 adapter factors applied on the fly (as in training)."""
    adapters = load_adapters(run)
    first = next(iter(adapters.values()))
    r = first["A"].shape[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    for p in model.parameters():
        p.requires_grad_(False)
    inject_adapters(model, r=r, alpha=first["scale"] * r, dropout=0.0, protect_k=None)
    with torch.no_grad():
        for name, entry in adapters.items():
            m = model.get_submodule(name)
            assert isinstance(m, LoRALinear)
            A, B, _ = effective_factors(entry)
            m.lora_A.copy_(A)
            m.lora_B.copy_(B)
    return _finish(model, device)