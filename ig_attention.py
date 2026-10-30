
# ig_attention.py
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def rolling_window(x: torch.Tensor, win: int) -> torch.Tensor:
    B, T = x.shape[:2]
    pad = win - 1
    if x.dim() == 3:
        D = x.size(-1)
        xpad = F.pad(x.transpose(1,2), (pad, 0), mode="constant", value=0.0).transpose(1,2)
        xs = []
        for k in range(win):
            xs.append(xpad[:, pad-k:pad-k+T, :])
        return torch.stack(xs[::-1], dim=2)
    else:
        xpad = F.pad(x, (pad, 0), mode="constant", value=0.0)
        xs = []
        for k in range(win):
            xs.append(xpad[:, pad-k:pad-k+T])
        return torch.stack(xs[::-1], dim=2)

def discretize(x: torch.Tensor, bins: int = 16) -> torch.Tensor:
    xclipped = x.clamp(-1.0, 1.0)
    idx = ((xclipped + 1.0) * 0.5 * bins).long()
    idx = idx.clamp(0, bins-1)
    return idx

def mutual_information_local(center_idx: torch.Tensor, neigh_idx: torch.Tensor, bins: int = 16, eps: float = 1e-6) -> torch.Tensor:
    B, T, win = neigh_idx.shape
    c_flat = center_idx.reshape(-1)
    n_flat = neigh_idx.reshape(-1)
    P_c = torch.bincount(c_flat, minlength=bins).float()
    P_c = (P_c + 1.0) / (P_c.sum() + bins)
    P_n = torch.bincount(n_flat, minlength=bins).float()
    P_n = (P_n + 1.0) / (P_n.sum() + bins)
    onehot_c = F.one_hot(c_flat, num_classes=bins).float()
    onehot_n = F.one_hot(n_flat, num_classes=bins).float()
    onehot_n_BT_win = onehot_n.view(B*T, win, bins).mean(dim=1)
    joint_counts = (onehot_c * onehot_n_BT_win).sum(dim=0)
    P_cn_diag = (joint_counts + 1.0) / (joint_counts.sum() + bins)
    ratio_diag = (P_cn_diag / (P_c * P_n + eps)).clamp(min=eps)
    r_match = ratio_diag[center_idx].unsqueeze(-1).expand(-1, -1, win)
    same = (neigh_idx == center_idx.unsqueeze(-1)).float()
    raw = same * torch.log(r_match + eps) + (1.0 - same) * torch.log(torch.full_like(r_match, 0.5))
    mu = raw.mean(dim=(0,1,2), keepdim=True)
    sd = raw.std(dim=(0,1,2), keepdim=True) + 1e-6
    z = (raw - mu) / sd
    miw = torch.sigmoid(z)
    return miw

@dataclass
class IGConfig:
    d_model: int = 128
    d_ff: int = 256
    win: int = 33
    bins: int = 16
    alpha: float = 1.5
    lam_dist: float = 0.01
    beta_energy: float = 0.01
    dropout: float = 0.0

class IGAttentionBlock(nn.Module):
    def __init__(self, cfg: IGConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor):
        B,T,D = x.shape
        win = self.cfg.win
        q = self.q(x)
        k = rolling_window(self.k(x), win)
        v = rolling_window(self.v(x), win)
        c_idx = discretize(x_raw, self.cfg.bins)
        n_idx = discretize(rolling_window(x_raw, win), self.cfg.bins)
        miw = mutual_information_local(c_idx, n_idx, bins=self.cfg.bins)
        dist = torch.arange(win, device=x.device).float()
        dist = (win - dist).unsqueeze(0).unsqueeze(0) / win
        q_exp = q.unsqueeze(2)
        dots = (q_exp * k).sum(-1) / (D ** 0.5)
        logits = dots + self.cfg.alpha * miw - self.cfg.lam_dist * (1.0 - dist)
        attn = torch.softmax(logits, dim=-1)
        energy = (attn * (1.0 / (miw + 1e-6))).mean()
        y = torch.einsum('btw,btwd->btd', attn, v)
        y = self.proj(y)
        y = self.drop(y)
        x = self.ln1(x + y)
        x2 = self.ff(x)
        x = self.ln2(x + x2)
        return x, energy * self.cfg.beta_energy

class LocalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, win: int, dropout: float = 0.0):
        super().__init__()
        self.win = win
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B,T,D = x.shape
        k = rolling_window(self.k(x), self.win)
        v = rolling_window(self.v(x), self.win)
        q = self.q(x).unsqueeze(2)
        dots = (q * k).sum(-1) / (D ** 0.5)
        attn = torch.softmax(dots, dim=-1)
        y = torch.einsum('btw,btwd->btd', attn, v)
        y = self.proj(y)
        y = self.drop(y)
        x = self.ln1(x + y)
        x2 = self.ff(x)
        x = self.ln2(x + x2)
        return x

class IGSeqModel(nn.Module):
    def __init__(self, cfg: IGConfig, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(1, cfg.d_model)
        self.blocks = nn.ModuleList([IGAttentionBlock(cfg) for _ in range(n_layers)])
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x_raw: torch.Tensor):
        x = self.embed(x_raw.unsqueeze(-1))
        energy_total = 0.0
        for blk in self.blocks:
            x, e = blk(x, x_raw)
            energy_total = energy_total + e
        y = self.head(x)
        return y.squeeze(-1), energy_total

class LocalSeqModel(nn.Module):
    def __init__(self, d_model: int = 128, d_ff: int = 256, win: int = 33, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([LocalAttentionBlock(d_model, d_ff, win, dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_raw: torch.Tensor):
        x = self.embed(x_raw.unsqueeze(-1))
        for blk in self.blocks:
            x = blk(x)
        y = self.head(x)
        return y.squeeze(-1)
