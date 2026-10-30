
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, label, gaussian_filter
import sys
import time
import os
import math
import dataclasses
from typing import List, Tuple

# -----------------------------
# Utilities
# -----------------------------

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def roll2(a, dy, dx):
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

# 8-neighborhood offsets
NEI = np.array([
    (-1,-1), (-1,0), (-1,1),
    ( 0,-1),          ( 0,1),
    ( 1,-1), ( 1,0),  ( 1,1)
], dtype=int)

# Laplacian kernel (4-neighbor)
LAPL_K = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)

# -----------------------------
# Model
# -----------------------------


@dataclass
class Params:
    H:int=128
    W:int=128
    K:int=8                      # alphabet size
    alpha:float=1.5              # "informational gravity" strength
    p_noise:float=0.01           # flip noise prob
    diffusion:float=0.2          # local mixing strength (0..1)
    expansion_decay:float=0.98   # weaken edges every T_exp steps
    T_exp:int=25                 # expansion period
    steps:int=500
    seed:int=42
    snap_every:int=50
    outdir:str='runs/demo'
    live:bool=False
    draw_every:int=5
    mi_window:int=7              # odd size for MI estimation window
    beta_cost:float=2.0          # softness of cost metric for rays/rotation
    rays:int=64                  # number of rays for lensing
    ray_steps:int=300            # steps per ray
    ray_step_size:float=1.0      # pixels per step
    gamma_deflect:float=0.2      # strength of deflection toward +grad(I)
    topk_clusters:int=3          # how many clusters to build rotation curves for
    ring_min:int=4               # min ring radius (pixels)
    ring_max:int=24              # max ring radius (pixels)
    ring_step:int=2              # ring step
    anti_mi:bool=False           # invert attraction
    shuffle_prob:float=0.0       # fraction to shuffle periodically
    shuffle_period:int=0         # period for shuffling (0 = off)
class InfoGravityCA:
    def __init__(self, params: Params):
        self.P = params
        rng = np.random.default_rng(self.P.seed)
        self.S = rng.integers(0, self.P.K, size=(self.P.H, self.P.W), dtype=np.int16)
        self.W = np.ones((self.P.H, self.P.W, 8), dtype=np.float32)
        self.rng = rng
        self.step_idx = 0

    def estimate_local_conditionals(self, window=5):
        k = window
        kernel = np.ones((k,k), dtype=np.float32)
        probs = np.empty((self.P.H, self.P.W, self.P.K), dtype=np.float32)
        total = float(k*k)
        for sym in range(self.P.K):
            mask = (self.S == sym).astype(np.float32)
            cnt = convolve(mask, kernel, mode='wrap')
            probs[..., sym] = (cnt + 1.0) / (total + self.P.K)
        return probs

    def info_map(self, probs):
        idx = (np.arange(self.P.H)[:,None], np.arange(self.P.W)[None,:], self.S)
        p_state = probs[idx]
        p_state = np.clip(p_state, 1e-6, 1.0)
        I = -np.log(p_state)
        return I.astype(np.float32)

    # ---- True local MI (center with each neighbor), estimated from window ----
    def local_mi_with_neighbors(self, window=None):
        if window is None:
            window = self.P.mi_window
        k = window
        half = k//2
        K = self.P.K

        # Prepare containers
        # MI per neighbor direction [H,W,8]
        MI = np.zeros((self.P.H, self.P.W, 8), dtype=np.float32)

        # Precompute one-hot volumes for states to speedup counting
        onehots = np.stack([(self.S==s).astype(np.float32) for s in range(K)], axis=-1)  # [H,W,K]

        # For each neighbor direction, compute joint distribution of (center, neighbor) in kxk window
        # Use convolutions to count occurrences
        kernel = np.ones((k,k), dtype=np.float32)
        # counts for center states in window
        C_counts = np.empty((self.P.H, self.P.W, K), dtype=np.float32)
        for s in range(K):
            C_counts[..., s] = convolve(onehots[..., s], kernel, mode='wrap')

        C_total = float(k*k)
        P_c = (C_counts + 1.0) / (C_total + K)  # Laplace

        # For each direction
        for di, (dy, dx) in enumerate(NEI):
            N_shift = roll2(self.S, dy, dx)
            N_onehots = np.stack([(N_shift==s).astype(np.float32) for s in range(K)], axis=-1)
            N_counts = np.empty((self.P.H, self.P.W, K), dtype=np.float32)
            for s in range(K):
                N_counts[..., s] = convolve(N_onehots[..., s], kernel, mode='wrap')
            P_n = (N_counts + 1.0) / (C_total + K)

            # Joint counts: for all pairs (c,n), need counts of both occurring in window
            # Approximation: product of convolved onehots of c (center) with convolved onehots of neighbor state in window
            # Better: convolve the AND map of equality? We'll do sum over symbols: J[c,n] ~ conv(1{S=c & S_shift=n})
            J_counts = np.zeros((self.P.H, self.P.W, K, K), dtype=np.float32)
            for c in range(K):
                for n in range(K):
                    both = ((self.S==c) & (N_shift==n)).astype(np.float32)
                    J_counts[..., c, n] = convolve(both, kernel, mode='wrap')
            P_cn = (J_counts + 1.0) / (C_total + K*K)

            # Compute MI = sum_{c,n} P(c,n) * log( P(c,n) / (P(c)P(n)) )
            # Avoid divide by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                denom = P_c[..., :, None] * P_n[..., None, :]
                ratio = np.clip(P_cn / np.clip(denom, 1e-9, 1.0), 1e-9, 1e6)
                mi = np.sum(P_cn * np.log(ratio), axis=(2,3))
            MI[..., di] = mi.astype(np.float32)

        # Normalize MI into a bounded range for stability
        # Scale by local mean/std
        m = np.mean(MI, axis=2, keepdims=True)
        s = np.std(MI, axis=2, keepdims=True) + 1e-6
        MI_norm = (MI - m) / s
        # Map to positive weights
        MI_pos = 1.0 / (1.0 + np.exp(-MI_norm))  # sigmoid -> (0,1)
        return MI_pos

    def attention_step(self, mi_weights):
        if self.P.anti_mi:
            miw = 1.0 - mi_weights
        else:
            miw = mi_weights
        logits = self.P.alpha * miw * self.W
        A = softmax(logits, axis=2)   # [H,W,8]
        votes = np.zeros((self.P.H, self.P.W, self.P.K), dtype=np.float32)
        for k, (dy,dx) in enumerate(NEI):
            Sj = roll2(self.S, dy, dx)
            votes[np.arange(self.P.H)[:,None], np.arange(self.P.W)[None,:], Sj] += A[..., k]
        votes[np.arange(self.P.H)[:,None], np.arange(self.P.W)[None,:], self.S] += self.P.diffusion
        S_new = np.argmax(votes, axis=2).astype(np.int16)
        return S_new

    def noise_step(self, S):
        flips = self.rng.random((self.P.H, self.P.W)) < self.P.p_noise
        if flips.any():
            S[flips] = self.rng.integers(0, self.P.K, size=flips.sum(), dtype=np.int16)
        return S

    def expansion_step(self):
        self.W *= self.P.expansion_decay
        jitter = self.rng.normal(loc=1.0, scale=0.01, size=self.W.shape).astype(np.float32)
        self.W *= np.clip(jitter, 0.8, 1.2)

    def laplacian(self, X):
        return convolve(X, LAPL_K, mode='wrap')

    def cluster_metrics(self, I, thresh_quantile=0.9):
        thr = np.quantile(I, thresh_quantile)
        mask = (I >= thr).astype(np.int32)
        structure = np.ones((3,3), dtype=int)  # 8-connectivity
        lbl, n = label(mask, structure=structure)
        return n, mask, lbl

    def step(self):
        probs = self.estimate_local_conditionals(window=5)
        I = self.info_map(probs)
        miw = self.local_mi_with_neighbors(window=self.P.mi_window)
        S_new = self.attention_step(miw)
        S_new = self.noise_step(S_new)
        if (self.step_idx % self.P.T_exp) == 0 and self.step_idx>0:
            self.expansion_step()
        self.S = S_new
        # Optional shuffling of a fraction of cells to break spurious structure
        if self.P.shuffle_period and (self.step_idx % self.P.shuffle_period == 0) and (self.P.shuffle_prob > 0):
            total = self.P.H * self.P.W
            n = int(total * self.P.shuffle_prob)
            if n > 0:
                ys = self.rng.integers(0, self.P.H, size=n)
                xs = self.rng.integers(0, self.P.W, size=n)
                self.S[ys, xs] = self.rng.integers(0, self.P.K, size=n, dtype=np.int16)
        self.step_idx += 1

        lapI = self.laplacian(I)
        neg_lapI = -lapI
        c = np.corrcoef(I.ravel(), neg_lapI.ravel())[0,1]
        nclusters, mask, lbl = self.cluster_metrics(I)
        return I, c, nclusters, mask, lbl, miw

# -----------------------------
# Lensing-like ray tracing
# -----------------------------

def simulate_rays(I: np.ndarray, P: Params, outdir: Path):
    H, W = I.shape
    # Smooth I for stable gradients
    Is = gaussian_filter(I, sigma=1.0)
    gy, gx = np.gradient(Is)  # note: row-major
    # Cost map from MI not available here; use Info as potential
    rays = []
    # Launch rays uniformly from left boundary with random y, heading right (vx=1, vy=0)
    rng = np.random.default_rng(123)
    for i in range(P.rays):
        y = rng.integers(0, H)
        x = 0
        vx, vy = 1.0, 0.0
        path = []
        for t in range(P.ray_steps):
            path.append((x, y))
            # deflect towards +grad(I)
            iy = int(y) % H
            ix = int(x) % W
            gx_loc = gx[iy, ix]
            gy_loc = gy[iy, ix]
            vx += P.gamma_deflect * gx_loc
            vy += P.gamma_deflect * gy_loc
            # normalize velocity
            vnorm = math.hypot(vx, vy) + 1e-9
            vx /= vnorm
            vy /= vnorm
            # advance
            x = (x + vx * P.ray_step_size) % W
            y = (y + vy * P.ray_step_size) % H
        rays.append(np.array(path))
    # Plot
    plt.figure()
    plt.imshow(Is, interpolation='nearest')
    for path in rays:
        plt.plot(path[:,0], path[:,1], linewidth=0.7)
    plt.title('Lensing-like rays over smoothed Info')
    plt.savefig(outdir/'rays.png')
    plt.close()
    # Simple aggregate: mean absolute deflection per ray (angle change)
    deflections = []
    for path in rays:
        # estimate angle changes
        angles = []
        # finite diffs
        diffs = np.diff(path, axis=0)
        for dv in diffs:
            ang = math.atan2(dv[1], dv[0])
            angles.append(ang)
        angles = np.unwrap(np.array(angles))
        total_defl = float(np.abs(angles[-1]-angles[0]))
        deflections.append(total_defl)
    np.save(outdir/'rays_deflections.npy', np.array(deflections))
    # plot histogram
    plt.figure()
    plt.hist(deflections, bins=20)
    plt.title('Distribution of total ray deflection')
    plt.xlabel('|Δangle| (radians)')
    plt.ylabel('count')
    plt.savefig(outdir/'rays_deflections_hist.png')
    plt.close()
    return float(np.mean(deflections)), float(np.std(deflections))

# -----------------------------
# Rotation curves
# -----------------------------

def extract_cluster_centers(mask: np.ndarray, labels: np.ndarray, topk:int=3):
    # pick largest components
    H, W = mask.shape
    ids, counts = np.unique(labels[labels>0], return_counts=True)
    if len(ids)==0:
        return []
    order = np.argsort(counts)[::-1]
    centers = []
    for idx in order[:topk]:
        lab = ids[idx]
        ys, xs = np.where(labels==lab)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        centers.append((cy, cx))
    return centers

def rotation_curves(I: np.ndarray, miw: np.ndarray, centers: List[Tuple[int,int]], P: Params, outdir: Path):
    # Effective cost map: lower cost where MI is high (sum over 8 dirs)
    MI_sum = np.sum(miw, axis=2)
    cost = 1.0 + np.exp(-P.beta_cost * (MI_sum - np.mean(MI_sum)))
    # Speed ~ 1/cost
    speed = 1.0 / cost

    curves = []  # list of (center, radii, speeds)
    H, W = I.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for (cy, cx) in centers:
        radii = list(range(P.ring_min, P.ring_max+1, P.ring_step))
        v_avg = []
        for r in radii:
            # circle mask approximated by thin ring
            dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
            ring = (np.abs(dist - r) <= 0.5)
            if ring.sum() < 10:
                v_avg.append(np.nan)
                continue
            v_avg.append(float(np.nanmean(speed[ring])))
        curves.append(((cy,cx), np.array(radii), np.array(v_avg, dtype=float)))

        # plot per center
        plt.figure()
        plt.plot(radii, v_avg)
        plt.title(f'Rotation curve (center=({cy},{cx}))')
        plt.xlabel('radius (px)')
        plt.ylabel('avg speed ~ 1/cost')
        plt.savefig(outdir/f'rotation_{cy:03d}_{cx:03d}.png')
        plt.close()

    # Overview image with centers
    plt.figure()
    plt.imshow(I, interpolation='nearest')
    for (cy,cx), _, _ in curves:
        plt.scatter([cx],[cy], s=30)
    plt.title('Centers used for rotation curves')
    plt.savefig(outdir/'rotation_centers.png')
    plt.close()
    return curves

# -----------------------------
# Live Viewer (unchanged from earlier, minimal integration)
# -----------------------------

class LiveViewer:
    def __init__(self, ca: InfoGravityCA):
        self.ca = ca
        self.enabled = ca.P.live
        self.backend_ok = True
        if not self.enabled:
            return
        try:
            plt.ion()
            self.fig_state, self.ax_state = plt.subplots(num='State', figsize=(5,5))
            self.im_state = self.ax_state.imshow(self.ca.S, interpolation='nearest')
            self.ax_state.set_title('State')

            self.fig_info, self.ax_info = plt.subplots(num='Info', figsize=(5,5))
            dummy_I = np.zeros_like(self.ca.S, dtype=float)
            self.im_info = self.ax_info.imshow(dummy_I, interpolation='nearest')
            self.ax_info.set_title('Info I')

            self.fig_metrics, self.ax_metrics = plt.subplots(num='Metrics', figsize=(6,4))
            self.corrs = []
            self.clusters = []
            self.line_corr, = self.ax_metrics.plot([], [], label='corr(I, -Lap(I))')
            self.line_clu, = self.ax_metrics.plot([], [], label='clusters')
            self.ax_metrics.legend(loc='best')
            self.ax_metrics.set_xlabel('step')
            self.ax_metrics.set_title('Metrics over time')
            plt.tight_layout()
            plt.pause(0.001)
        except Exception as e:
            print(f"[Live] GUI backend not available ({e}). Falling back to headless saving.", file=sys.stderr)
            self.backend_ok = False
            self.enabled = False

    def update(self, t, I, corr, ncl):
        if not (self.enabled and self.backend_ok):
            return
        try:
            self.im_state.set_data(self.ca.S)
            self.ax_state.set_title(f'State t={t}')
            self.im_info.set_data(I)
            self.ax_info.set_title(f'Info I t={t}')
            self.corrs.append(corr)
            self.clusters.append(ncl)
            xs = np.arange(len(self.corrs))
            self.line_corr.set_data(xs, self.corrs)
            self.line_clu.set_data(xs, self.clusters)
            self.ax_metrics.relim()
            self.ax_metrics.autoscale_view()
            self.fig_state.canvas.flush_events()
            self.fig_info.canvas.flush_events()
            self.fig_metrics.canvas.flush_events()
            plt.pause(0.001)
        except Exception as e:
            print(f"[Live] Update failed: {e}", file=sys.stderr)
            self.enabled = False

# -----------------------------
# HTML report generator
# -----------------------------

def html_report(outdir: Path, summary: dict):
    html_path = outdir/'report.html'
    esc = lambda s: str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    imgs = [
        'state_{:04d}.png'.format(max(0, summary.get("last_snap_step", 0))),
        'info_{:04d}.png'.format(max(0, summary.get("last_snap_step", 0))),
        'mask_{:04d}.png'.format(max(0, summary.get("last_snap_step", 0))),
        'corr_time.png',
        'clusters_time.png',
        'rays.png',
        'rays_deflections_hist.png',
        'rotation_centers.png'
    ]
    # plus rotation curve images (glob)
    rots = [p for p in os.listdir(outdir) if p.startswith('rotation_') and p.endswith('.png')]
    body_imgs = ''.join([f'<div><img src="{esc(img)}" style="max-width:100%;"></div>\n' for img in imgs if (outdir/img).exists()])
    body_imgs += ''.join([f'<div><img src="{esc(img)}" style="max-width:100%;"></div>\n' for img in rots])

    # summary stats
    s = summary
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Info-Gravity CA Report</title>
<style> body{{font-family: -apple-system, Segoe UI, sans-serif; margin:20px;}} h1,h2{{margin-top:1.2em}}</style>
</head><body>
<h1>Informational Gravity CA — Report</h1>
<p><b>Steps:</b> {esc(s.get("steps"))} |
<b>alpha:</b> {esc(s.get("alpha"))} |
<b>mi_window:</b> {esc(s.get("mi_window"))} |
<b>rays:</b> {esc(s.get("rays"))} |
<b>ray_steps:</b> {esc(s.get("ray_steps"))}</p>

<h2>Key Metrics</h2>
<ul>
<li>Anti-MI: <b>{esc(s.get('anti_mi'))}</b>; Shuffle: <b>{esc(s.get('shuffle_prob'))}</b> every <b>{esc(s.get('shuffle_period'))}</b> steps</li>
<li>Mean corr(I, -Lap(I)): <b>{esc(s.get("mean_corr"))}</b> (std {esc(s.get("std_corr"))})</li>
<li>Mean high-I clusters: <b>{esc(s.get("mean_clusters"))}</b> (std {esc(s.get("std_clusters"))})</li>
<li>Mean ray deflection: <b>{esc(s.get("mean_deflection"))}</b> (std {esc(s.get("std_deflection"))})</li>
</ul>

<h2>Visuals</h2>
{body_imgs}

</body></html>"""
    Path(html_path).write_text(html, encoding='utf-8')
    return html_path

# -----------------------------
# Runner
# -----------------------------

def run_experiment(P:Params):
    outdir = Path(P.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ca = InfoGravityCA(P)
    viewer = LiveViewer(ca)

    corrs = []
    cl_counts = []
    last_snap = 0

    for t in range(P.steps):
        I, corr, ncl, mask, lbl, miw = ca.step()
        corrs.append(float(corr))
        cl_counts.append(int(ncl))

        if P.live and (t % P.draw_every == 0):
            viewer.update(t, I, corr, ncl)

        if (t % P.snap_every)==0 or t==(P.steps-1):
            last_snap = t
            plt.figure()
            plt.imshow(ca.S, interpolation='nearest')
            plt.title(f'State t={t}')
            plt.savefig(outdir/f'state_{t:04d}.png')
            plt.close()

            plt.figure()
            plt.imshow(I, interpolation='nearest')
            plt.title(f'Info I t={t}')
            plt.colorbar()
            plt.savefig(outdir/f'info_{t:04d}.png')
            plt.close()

            plt.figure()
            plt.imshow(mask, interpolation='nearest')
            plt.title(f'High-I mask t={t}')
            plt.savefig(outdir/f'mask_{t:04d}.png')
            plt.close()

    xs = np.arange(P.steps)
    plt.figure()
    plt.plot(xs, corrs)
    plt.title('corr(I, -Lap(I)) over time')
    plt.xlabel('step')
    plt.ylabel('correlation')
    plt.savefig(outdir/'corr_time.png')
    plt.close()

    plt.figure()
    plt.plot(xs, cl_counts)
    plt.title('High-I connected components over time')
    plt.xlabel('step')
    plt.ylabel('components')
    plt.savefig(outdir/'clusters_time.png')
    plt.close()

    np.savez(outdir/'metrics.npz', corrs=np.array(corrs), clusters=np.array(cl_counts))

    # ---- Lensing-like rays on final I ----
    mean_defl, std_defl = simulate_rays(I, P, outdir)

    # ---- Rotation curves on final frame ----
    centers = extract_cluster_centers(mask, lbl, topk=P.topk_clusters)
    curves = rotation_curves(I, miw, centers, P, outdir)

    # Compute flatness (slope of last third) for first curve if exists
    rot_slopes = []
    for (_center, radii, v_avg) in curves:
        if len(radii) >= 6 and np.isfinite(v_avg).sum() >= 6:
            # take last third points
            m = max(3, len(radii)//3)
            x = radii[-m:]
            y = v_avg[-m:]
            # remove NaNs
            mask = np.isfinite(y)
            if mask.sum() >= 3:
                x2 = np.array(x)[mask]
                y2 = np.array(y)[mask]
                A = np.vstack([x2, np.ones_like(x2)]).T
                slope, _ = np.linalg.lstsq(A, y2, rcond=None)[0]
                rot_slopes.append(float(slope))
    # ---- HTML report ----
    summary = {
        "steps": P.steps,
        "alpha": P.alpha,
        "mi_window": P.mi_window,
        "rays": P.rays,
        "ray_steps": P.ray_steps,
        "mean_corr": float(np.mean(corrs)),
        "std_corr": float(np.std(corrs)),
        "mean_clusters": float(np.mean(cl_counts)),
        "std_clusters": float(np.std(cl_counts)),
        "mean_deflection": mean_defl,
        "std_deflection": std_defl,
        "last_snap_step": last_snap,
        "rot_slopes": rot_slopes,
        "anti_mi": P.anti_mi,
        "shuffle_prob": P.shuffle_prob,
        "shuffle_period": P.shuffle_period
    }
    # persist summary for suite aggregator
    import json
    summary['outdir'] = str(outdir)
    (outdir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    html_path = html_report(outdir, summary)
    return html_path

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=int, default=128)
    ap.add_argument('--W', type=int, default=128)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--p_noise', type=float, default=0.01)
    ap.add_argument('--diffusion', type=float, default=0.2)
    ap.add_argument('--expansion_decay', type=float, default=0.98)
    ap.add_argument('--T_exp', type=int, default=25)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--snap_every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', type=str, default='runs/demo')
    ap.add_argument('--live', action='store_true', help='enable real-time visualization')
    ap.add_argument('--draw_every', type=int, default=5, help='update live plots every N steps')
    ap.add_argument('--mi_window', type=int, default=7)
    ap.add_argument('--beta_cost', type=float, default=2.0)
    ap.add_argument('--rays', type=int, default=64)
    ap.add_argument('--ray_steps', type=int, default=300)
    ap.add_argument('--ray_step_size', type=float, default=1.0)
    ap.add_argument('--gamma_deflect', type=float, default=0.2)
    ap.add_argument('--topk_clusters', type=int, default=3)
    ap.add_argument('--ring_min', type=int, default=4)
    ap.add_argument('--ring_max', type=int, default=24)
    ap.add_argument('--ring_step', type=int, default=2)
    ap.add_argument('--anti_mi', action='store_true')
    ap.add_argument('--shuffle_prob', type=float, default=0.0)
    ap.add_argument('--shuffle_period', type=int, default=0)
    ap.add_argument('--suite', action='store_true', help='run predefined experiment suite and master report')

    args = ap.parse_args()
    P = Params(
        H=args.H, W=args.W, K=args.K, alpha=args.alpha, p_noise=args.p_noise,
        diffusion=args.diffusion, expansion_decay=args.expansion_decay,
        T_exp=args.T_exp, steps=args.steps, seed=args.seed, snap_every=args.snap_every,
        outdir=args.outdir, live=args.live, draw_every=args.draw_every,
        mi_window=args.mi_window, beta_cost=args.beta_cost,
        rays=args.rays, ray_steps=args.ray_steps, ray_step_size=args.ray_step_size,
        gamma_deflect=args.gamma_deflect, topk_clusters=args.topk_clusters,
        ring_min=args.ring_min, ring_max=args.ring_max, ring_step=args.ring_step,
        anti_mi=args.anti_mi, shuffle_prob=args.shuffle_prob, shuffle_period=args.shuffle_period
    )

    if args.suite:
        # Run predefined suite
        master_dir = Path(P.outdir)
        configs = []
        # Phase A: alpha scan
        for a in [0.0, 1.2, 1.8]:
            Q = P
            Q = dataclasses.replace(Q, alpha=a, outdir=str(master_dir/f'A_alpha_{a}'))
            configs.append(Q)
        # Phase B: falsification
        configs.append(dataclasses.replace(P, alpha=1.4, anti_mi=True, outdir=str(master_dir/'B_anti_mi')))
        configs.append(dataclasses.replace(P, alpha=1.4, shuffle_prob=0.05, shuffle_period=20, outdir=str(master_dir/'B_shuffle')))
        # Phase C: reproducibility (3 seeds)
        for sd in [41,42,43]:
            configs.append(dataclasses.replace(P, alpha=1.8, seed=sd, outdir=str(master_dir/f'C_seed_{sd}')))

        summaries = []
        reports = []
        for cfg in configs:
            print(f'[SUITE] Running {cfg.outdir} ...')
            html = run_experiment(cfg)
            # Load summary.json written by run_experiment
            import json
            with open(Path(cfg.outdir)/'summary.json','r') as f:
                summaries.append(json.load(f))
            reports.append(html)

        # Master report
        master = master_dir/'master_report.html'
        rows = []
        for s in summaries:
            rows.append(f"<tr><td>{s.get('outdir','')}</td><td>{s.get('alpha')}</td><td>{s.get('anti_mi')}</td><td>{s.get('shuffle_prob')}</td><td>{s.get('mean_corr'):.4f}</td><td>{s.get('mean_deflection'):.4f}</td><td>{np.mean(s.get('rot_slopes') or [np.nan]):.4f}</td><td><a href='{Path(s.get('outdir',''))/'report.html'}'>report</a></td></tr>")
        table = """
        <table border='1' cellpadding='6' cellspacing='0'>
        <tr><th>run</th><th>alpha</th><th>anti_mi</th><th>shuffle</th><th>mean corr</th><th>mean deflection</th><th>rot slope (mean)</th><th>link</th></tr>
        %s
        </table>
        """ % ("\n".join(rows))
        master_html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Experiment Suite</title>
        <style>body{{font-family:-apple-system,Segoe UI,sans-serif;margin:20px}}</style></head><body>
        <h1>Experiment Suite — Informational Gravity</h1>
        {table}
        </body></html>"""
        Path(master).write_text(master_html, encoding='utf-8')
        print(f"Master report: {master}")
    else:
        out_html = run_experiment(P)
        print(f"Report written to: {out_html}")
