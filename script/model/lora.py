import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, freeze_base: bool = True):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), "LoRALinear expects nn.Linear"

        # base
        self.base = base_linear
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

        self.in_f = self.base.in_features
        self.out_f = self.base.out_features
        self.r = int(r)
        self.scaling = float(alpha) / float(r) if r > 0 else 0.0
        self.use_lora = (self.r > 0)
        self.merged = False

        if self.use_lora:
            self.lora_A = nn.Parameter(torch.empty(self.r, self.in_f,
                                                   device=self.base.weight.device,
                                                   dtype=self.base.weight.dtype))
            self.lora_B = nn.Parameter(torch.empty(self.out_f, self.r,
                                                   device=self.base.weight.device,
                                                   dtype=self.base.weight.dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

    def forward(self, x):
        if (not self.use_lora) or self.merged:
            return self.base(x)
        
        out = self.base(x)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + self.scaling * delta
    
    @torch.no_grad()
    def merge(self):
        if (not self.use_lora) or self.merged:
            self.merged = True
            return
        update = (self.lora_B @ self.lora_A) * self.scaling   # [out_f, in_f]
        self.base.weight.add_(update)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if (not self.use_lora) or (not self.merged):
            self.merged = False
            return
        update = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.sub_(update)
        self.merged = False
    
@torch.no_grad()    
def _wrap_attr_with_lora(module: nn.Module, attr: str,
                         r: int, alpha: int, dropout: float, freeze_base: bool):
    lin = getattr(module, attr)
    assert isinstance(lin, nn.Linear), f"{attr} isn't nn.Linear"
    setattr(module, attr, LoRALinear(lin, r=r, alpha=alpha, dropout=dropout, freeze_base=freeze_base))

@torch.no_grad()
def inject_lora_into_pairawareblock(
    block: nn.Module,
    cfg_per_attn: Dict[str, Tuple[int, int]],  # {attn_name: (r, alpha)}
    dropout: float = 0.0,
    freeze_base: bool = True,
    proj_names: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
):
    for attn_name, (r, alpha) in cfg_per_attn.items():
        attn = getattr(block, attn_name, None)
        if attn is None:
            continue
        for pn in proj_names:
            if hasattr(attn, pn):
                _wrap_attr_with_lora(attn, pn, r=r, alpha=alpha, dropout=dropout, freeze_base=freeze_base)

@torch.no_grad()
def inject_lora_into_trunk_last_n(
    module: nn.Module,
    last_n: int = 2,
    cfg_seq_pair: Tuple[Tuple[int,int], Tuple[int,int]] = ((8,16),(4,8)),
    dropout: float = 0.1,
    freeze_base: bool = True,
):
    assert hasattr(module, "pair_aware_trunk") and isinstance(module.pair_aware_trunk, nn.ModuleList)
    n = len(module.pair_aware_trunk)
    assert last_n <= n

    (r_seq, a_seq), (r_pair, a_pair) = cfg_seq_pair
    cfg = {
        "seq_attn": (r_seq, a_seq),
        "tri_attn_start": (r_pair, a_pair),
        "tri_attn_end": (r_pair, a_pair),
    }
    for idx in range(n - last_n, n):
        inject_lora_into_pairawareblock(
            module.pair_aware_trunk[idx],
            cfg_per_attn=cfg,
            dropout=dropout,
            freeze_base=freeze_base,
        )

@torch.no_grad()
def freeze_module_except_lora(module: nn.Module):
    for n, p in module.named_parameters():
        is_lora = (".A.weight" in n) or (".B.weight" in n) or (".lora_A" in n) or (".lora_B" in n)
        if is_lora:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

def collect_trainable_module_names(module: nn.Module):
    mods = set()
    for n, p in module.named_parameters():
        if p.requires_grad:
            parts = n.split(".")
            mods.add(".".join(parts[:-1]) if len(parts) > 1 else "(root)")
    return sorted(mods)

def collect_trainable_param_names(module: nn.Module):
    return [n for n, p in module.named_parameters() if p.requires_grad]

def print_trainable(model: nn.Module):
    mods = collect_trainable_module_names(model)
    params = collect_trainable_param_names(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Trainable Modules] ({len(mods)}):")
    for m in mods:
        print("  -", m)
    # print(f"[Counts] trainable={trainable:,} / total={total:,} ({trainable/max(1,total):.4%})")
    # for n in params: print("  *", n)

def build_model_with_lora(model, last_n, cfg_seq_pair=((8,16),(4,8)), dropout=0.1, 
                          freeze_base=True, print_trainabel=False):
    inject_lora_into_trunk_last_n(
        model.peptide_encoder.pair_aware_trunk, last_n,
        cfg_seq_pair, dropout, freeze_base
    )
    freeze_module_except_lora(model.peptide_encoder)
    freeze_module_except_lora(model.cdr3_encoder)
    if print_trainabel:
        print_trainable(model)
    return model

######## 
def iter_lora_linear(module: nn.Module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m

@torch.no_grad()
def merge_model_lora(model: nn.Module):
    for m in iter_lora_linear(model):
        m.merge()

@torch.no_grad()
def unmerge_model_lora(model: nn.Module):
    for m in iter_lora_linear(model):
        m.unmerge()

def lora_state_dict(model: nn.Module):
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear) and m.use_lora:
            sd[f"{name}.lora_A"] = m.lora_A
            sd[f"{name}.lora_B"] = m.lora_B
    return sd

@torch.no_grad()
def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], strict: bool = False):
    found = 0
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear) and m.use_lora:
            kA, kB = f"{name}.lora_A", f"{name}.lora_B"
            if kA in state and kB in state:
                m.lora_A.copy_(state[kA])
                m.lora_B.copy_(state[kB])
                found += 1
            elif strict:
                raise KeyError(f"Missing {kA}/{kB} in provided state dict")
    return found
