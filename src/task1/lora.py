import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        dout, din = base.weight.shape
        dev = base.weight.device
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, din, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(dout, r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.register_buffer("U", None, persistent=False)
        self.register_buffer("V", None, persistent=False)

    @torch.no_grad()
    def set_protection(self, k: int):
        with torch.autocast("cuda", enabled=False):
            U, _, Vh = torch.linalg.svd(self.base.weight.float(), full_matrices=False)
        self.U = U[:, :k].contiguous()
        self.V = Vh[:k].T.contiguous()

    def factors(self):
        """Effective (projected) factors. Projection done in FP32, never as dense matrices."""
        A, B = self.lora_A, self.lora_B
        if self.U is not None:
            with torch.autocast("cuda", enabled=False):
                B = B - self.U @ (self.U.T @ B)
                A = A - (A @ self.V) @ self.V.T
        return A, B

    def forward(self, x):
        A, B = self.factors()
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), A), B)


def inject_adapters(model, r, alpha, dropout, protect_k=None) -> dict:
    """Wrap every target projection in every layer. Call set_seed() just before this."""
    mods = []
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in TARGETS:
                if hasattr(parent, name):
                    m = LoRALinear(getattr(parent, name), r, alpha, dropout)
                    setattr(parent, name, m)
                    mods.append(m)
    info = {"adapter_modules": len(mods)}
    if protect_k:
        torch.cuda.synchronize()
        t0 = time.time()
        for m in mods:
            m.set_protection(protect_k)
        torch.cuda.synchronize()
        info["basis_preprocessing_s"] = round(time.time() - t0, 2)
        info["basis_storage_mb"] = round(
            sum((m.U.numel() + m.V.numel()) * 4 for m in mods) / 1024**2, 1)
    return info


@torch.no_grad()
def export_adapters(model, out_dir) -> dict:
    """Save raw factors; for OPLoRA also the projected factors used for inference."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out, resid = {}, []
    for name, m in model.named_modules():
        if not isinstance(m, LoRALinear):
            continue
        entry = {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu(), "scale": m.scale}
        if m.U is not None:
            A, B = m.factors()
            entry["A_proj"], entry["B_proj"] = A.cpu(), B.cpu()
            resid.append(max(float((m.U.T @ B).abs().max()), float((A @ m.V).abs().max())))
        out[name] = entry
    torch.save(out, out_dir / "adapters.pt")
    return {"max_projection_residual": max(resid)} if resid else {}