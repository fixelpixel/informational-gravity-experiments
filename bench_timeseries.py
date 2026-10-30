
# bench_timeseries.py
import time, math, argparse, json
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ig_attention import IGConfig, IGSeqModel, LocalSeqModel, get_device

def make_dataset(n_series: int, T: int, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(n_series):
        x = np.zeros(T, dtype=np.float32)
        phi = 0.6 + 0.2*rng.random()
        val = rng.normal(scale=0.2)
        freq = rng.uniform(0.01, 0.05)
        phase = rng.uniform(0, 2*np.pi)
        for t in range(T):
            val = phi*val + rng.normal(scale=0.15)
            x[t] = val + 0.5*np.sin(2*np.pi*freq*t + phase)
        for _ in range(rng.integers(2,5)):
            pos = rng.integers(20, T-20)
            length = rng.integers(8, 16)
            amp = rng.uniform(0.5, 0.9) * rng.choice([-1,1])
            x[pos:pos+length] += amp * np.hanning(length)
        x = np.tanh(x)
        X.append(x)
    X = np.stack(X, axis=0)
    return torch.tensor(X, dtype=torch.float32)

def make_batches(X: torch.Tensor, ctx: int, bs: int, device):
    N, T = X.shape
    xs, ys = [], []
    for n in range(N):
        for t in range(ctx, T-1):
            xs.append(X[n, t-ctx:t].unsqueeze(0))
            ys.append(X[n, t+1].unsqueeze(0))
    Xb = torch.cat(xs, dim=0)
    Yb = torch.cat(ys, dim=0)
    perm = torch.randperm(Xb.size(0))
    Xb = Xb[perm]
    Yb = Yb[perm]
    m = (Xb.size(0) // bs) * bs
    Xb = Xb[:m].to(device)
    Yb = Yb[:m].to(device)
    return Xb, Yb

def train_epoch_local(model, opt, Xb, Yb, bs):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    t0 = time.time()
    for i in range(0, Xb.size(0), bs):
        xb = Xb[i:i+bs]
        yb = Yb[i:i+bs]
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred[:, -1], yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    dt = time.time() - t0
    return total_loss / (Xb.size(0)//bs), dt

def train_epoch_ig(model, opt, Xb, Yb, bs):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    energy_acc = 0.0
    t0 = time.time()
    for i in range(0, Xb.size(0), bs):
        xb = Xb[i:i+bs]
        yb = Yb[i:i+bs]
        opt.zero_grad(set_to_none=True)
        pred, energy = model(xb)
        loss = loss_fn(pred[:, -1], yb) + energy
        loss.backward()
        opt.step()
        total_loss += loss.item()
        energy_acc += float(energy.detach().cpu())
    dt = time.time() - t0
    return total_loss / (Xb.size(0)//bs), dt, energy_acc / (Xb.size(0)//bs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--series', type=int, default=64)
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--ctx', type=int, default=128)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--win', type=int, default=33)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--d_ff', type=int, default=256)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--lam_dist', type=float, default=0.01)
    ap.add_argument('--beta_energy', type=float, default=0.01)
    ap.add_argument('--bins', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--outdir', type=str, default='runs_ig_ts')
    args = ap.parse_args()

    device = get_device() if args.device=='auto' else torch.device(args.device)
    print('Device:', device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X = make_dataset(args.series, args.T, seed=args.seed)
    Xb, Yb = make_batches(X, args.ctx, args.bs, device)

    local = LocalSeqModel(d_model=args.d_model, d_ff=args.d_ff, win=args.win, n_layers=args.layers).to(device)
    from ig_attention import IGConfig, IGSeqModel
    igcfg = IGConfig(d_model=args.d_model, d_ff=args.d_ff, win=args.win, bins=args.bins,
                     alpha=args.alpha, lam_dist=args.lam_dist, beta_energy=args.beta_energy)
    ig = IGSeqModel(igcfg, n_layers=args.layers).to(device)

    opt_local = optim.AdamW(local.parameters(), lr=args.lr)
    opt_ig = optim.AdamW(ig.parameters(), lr=args.lr)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    hist = {"local": {"loss": [], "sec": []}, "ig": {"loss": [], "sec": [], "energy": []}}

    for ep in range(1, args.epochs+1):
        l_loss, l_sec = train_epoch_local(local, opt_local, Xb, Yb, args.bs)
        ig_loss, ig_sec, ig_energy = train_epoch_ig(ig, opt_ig, Xb, Yb, args.bs)
        hist["local"]["loss"].append(l_loss); hist["local"]["sec"].append(l_sec)
        hist["ig"]["loss"].append(ig_loss); hist["ig"]["sec"].append(ig_sec); hist["ig"]["energy"].append(ig_energy)
        print(f"Epoch {ep:02d}: Local loss={l_loss:.4f} ({l_sec:.2f}s) | IG loss={ig_loss:.4f} ({ig_sec:.2f}s) energy={ig_energy:.4f}")

    with open(outdir/'results.json', 'w') as f: json.dump(hist, f, indent=2)
    torch.save({"hist": hist}, outdir/'checkpoint.pt')
    print('Saved to', outdir)

if __name__ == '__main__':
    main()
