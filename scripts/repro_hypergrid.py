#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal reproducible experiment on an enumerable HyperGrid.

Verifies two claims made in this repo's insight reports:

1. Systematic underfitting (T07 / T01): is the regression slope of
   log P_T(x) on log R(x) below 1.0?
2. Metric failure (T32): do Spearman correlation, mean-reward ratio and
   mode count give a flattering score to a model whose TV error is large?

numpy only. Single hidden layer MLP with hand-written backprop, runs on CPU.
The point is that the claims can be checked by hand, not that this is fast.

Usage:
    python3 scripts/repro_hypergrid.py
    python3 scripts/repro_hypergrid.py --H 12 --iters 6000
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

# force UTF-8 output so CJK survives pipes/redirects
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def make_reward(H, D, R0=1e-3, R1=0.5, R2=2.0):
    """Concentric-shell multimodal reward from Bengio et al. 2021 (T01).

    R0 controls difficulty: the smaller it is, the harder to cross the
    low-reward region between modes.
    """
    coords = np.array(list(itertools.product(range(H), repeat=D)), dtype=np.float64)
    z = np.abs(coords / (H - 1) - 0.5)
    shell1 = np.all(z > 0.25, axis=1)
    shell2 = np.all((z > 0.3) & (z < 0.4), axis=1)
    return coords.astype(int), R0 + R1 * shell1 + R2 * shell2


def sidx(s, H, D):
    i = 0
    for v in s:
        i = i * H + int(v)
    return i


class Policy:
    """Forward policy P_F(a|s) plus the learnable scalar logZ used by TB."""

    def __init__(self, H, D, hidden=128, seed=0):
        self.H, self.D = H, D
        rng = np.random.default_rng(seed)
        din, dout = H * D, D + 1
        self.W1 = rng.normal(0, math.sqrt(2.0 / din), (din, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, math.sqrt(2.0 / hidden), (hidden, dout))
        self.b2 = np.zeros(dout)
        self.logZ = np.zeros(1)
        self._cache = None

    def encode(self, states):
        B = states.shape[0]
        x = np.zeros((B, self.H * self.D))
        for d in range(self.D):
            x[np.arange(B), d * self.H + states[:, d]] = 1.0
        return x

    def forward(self, states):
        x = self.encode(states)
        hp = x @ self.W1 + self.b1
        h = np.maximum(hp, 0.0)
        self._cache = (x, hp, h)
        return h @ self.W2 + self.b2

    @staticmethod
    def log_softmax(logits, mask):
        z = np.where(mask, -1e30, logits)
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    def apply_grads(self, dlogits, lr):
        """dlogits is dLoss/dlogits. Descends the gradient."""
        x, hp, h = self._cache
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dhp = (dlogits @ self.W2.T) * (hp > 0)
        dW1 = x.T @ dhp
        db1 = dhp.sum(axis=0)
        for p, g in ((self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2)):
            np.clip(g, -1.0, 1.0, out=g)
            p -= lr * g


def sample_batch(pol, batch, rng, eps=0.0):
    """Sample complete trajectories. Backward policy is uniform over parents,
    so log P_B is computed analytically."""
    H, D = pol.H, pol.D
    s = np.zeros((batch, D), dtype=int)
    alive = np.ones(batch, dtype=bool)
    logpf = np.zeros(batch)
    logpb = np.zeros(batch)
    steps = np.zeros(batch, dtype=int)
    tape = []

    for _ in range(D * (H - 1) + 1):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        cur = s[idx]
        logits = pol.forward(cur)
        at_edge = cur >= H - 1
        mask = np.concatenate([at_edge, np.zeros((len(idx), 1), bool)], axis=1)
        logp = pol.log_softmax(logits, mask)
        p = np.exp(logp)
        if eps > 0:
            legal = (~mask).astype(float)
            u = legal / legal.sum(axis=1, keepdims=True)
            p = (1 - eps) * p + eps * u
            p /= p.sum(axis=1, keepdims=True)
        act = (rng.random((len(idx), 1)) > p.cumsum(axis=1)).sum(axis=1)
        act = np.minimum(act, D)
        logpf[idx] += logp[np.arange(len(idx)), act]
        tape.append((cur.copy(), mask.copy(), act.copy(), idx.copy()))

        stop = act == D
        move = ~stop
        if move.any():
            mi = idx[move]
            s[mi, act[move]] += 1
            steps[mi] += 1
            logpb[mi] -= np.log(np.maximum((s[mi] > 0).sum(axis=1), 1))
        alive[idx[stop]] = False
    return s, logpf, logpb, steps, tape


def tb_step(pol, batch, rng, R, H, D, lr, eps):
    """One Trajectory Balance update.

    L = ( logZ + log P_F(tau) - log R(x) - log P_B(tau|x) )^2
    dL/d(log P_F) = 2*delta, so dL/dlogits = 2*delta * (onehot(act) - p).
    """
    s, logpf, logpb, steps, tape = sample_batch(pol, batch, rng, eps)
    logR = np.log(R[np.array([sidx(x, H, D) for x in s])])
    delta = pol.logZ[0] + logpf - logR - logpb
    loss = float((delta ** 2).mean())

    pol.logZ -= lr * np.clip(2.0 * delta.mean(), -5.0, 5.0)

    coef = 2.0 * delta / batch
    for cur, mask, act, idx in reversed(tape):
        logits = pol.forward(cur)
        p = np.exp(pol.log_softmax(logits, mask))
        g = -p
        g[np.arange(len(idx)), act] += 1.0        # d(logp[act])/dlogits
        dlogits = coef[idx][:, None] * g          # dL/dlogits
        dlogits[mask] = 0.0
        pol.apply_grads(dlogits, lr)
    return loss, float(steps.mean())


def terminal_dist(pol, n, rng, H, D, chunk=2000):
    counts = np.zeros(H ** D)
    got = 0
    while got < n:
        b = min(chunk, n - got)
        s, _, _, _, _ = sample_batch(pol, b, rng, eps=0.0)
        for x in s:
            counts[sidx(x, H, D)] += 1
        got += b
    return counts / counts.sum()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--eval-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="artifacts/repro_hypergrid.json")
    args = ap.parse_args()

    H, D = args.H, args.D
    rng = np.random.default_rng(args.seed)
    _, R = make_reward(H, D)
    Z_true = R.sum()
    P_star = R / Z_true
    n_modes = int((R > 1.0).sum())

    pol = Policy(H, D, seed=args.seed)
    print(f"环境：{D} 维 HyperGrid，边长 {H}，共 {H**D} 个终止态")
    print(f"真实 Z = {Z_true:.4f}，log Z = {math.log(Z_true):.4f}")
    print(f"奖励范围 [{R.min():.4f}, {R.max():.3f}]，高奖励模态（R>1）{n_modes} 个\n")

    curve = []
    every = max(1, args.iters // 10)
    for it in range(1, args.iters + 1):
        loss, mlen = tb_step(pol, args.batch, rng, R, H, D, args.lr, args.eps)
        if it == 1 or it % every == 0:
            Pt = terminal_dist(pol, 4000, rng, H, D)
            tv = float(0.5 * np.abs(Pt - P_star).sum())
            curve.append({"iter": it, "loss": loss, "tv": tv,
                          "logZ": float(pol.logZ[0]), "mean_len": mlen})
            print(f"iter {it:>5} | TB loss {loss:>8.4f} | TV {tv:.4f} | "
                  f"logZ {pol.logZ[0]:>6.3f} | 平均轨迹长 {mlen:.2f}")

    print("\n" + "=" * 76)
    Pt = terminal_dist(pol, args.eval_samples, rng, H, D)
    tv = float(0.5 * np.abs(Pt - P_star).sum())
    l1 = float(np.abs(Pt - P_star).sum())

    m = Pt > 0
    lp, lR = np.log(Pt[m]), np.log(R[m])
    A = np.vstack([lR, np.ones_like(lR)]).T
    slope, intercept = np.linalg.lstsq(A, lp, rcond=None)[0]
    resid = lp - A @ np.array([slope, intercept])
    sst = ((lp - lp.mean()) ** 2).sum()
    r2 = float(1 - (resid ** 2).sum() / sst) if sst > 0 else float("nan")

    sp = spearman(Pt, R)
    mrr = float((Pt * R).sum() / (P_star * R).sum())
    thr = 1.0 / (H ** D) / 10
    modes_found = int(((R > 1.0) & (Pt > thr)).sum())

    print("【论断 1】系统性欠拟合（T07 / T01）")
    print(f"  log P_T(x) 对 log R(x) 回归斜率 = {slope:.3f}（理想 1.0），R² = {r2:.3f}")
    verdict = "复现：斜率 < 1，模型系统性低估高奖励对象" if slope < 0.95 else "本次未复现"
    print(f"  {verdict}（T01 在小分子任务上报的斜率是 0.58）")
    print(f"  学到 logZ = {pol.logZ[0]:.3f}，真实 {math.log(Z_true):.3f}，"
          f"偏差 {abs(pol.logZ[0] - math.log(Z_true)):.3f}")

    print("\n【论断 2】流行指标是否反映分布正确性（T32）")
    print(f"  真实分布误差   TV = {tv:.4f}    L1 = {l1:.4f}   <- 唯一可信基准")
    print(f"  Spearman 相关  {sp:+.4f}                       <- 只看排序")
    print(f"  平均奖励比     {mrr:.4f}                        <- 越接近 1 越像学对了")
    print(f"  发现模态数     {modes_found}/{n_modes}                            <- 只反映搜索能力")

    res = {
        "env": {"H": H, "D": D, "n_terminal": H ** D, "Z_true": float(Z_true),
                "log_Z_true": math.log(Z_true), "n_modes": n_modes},
        "train": {"iters": args.iters, "batch": args.batch, "lr": args.lr,
                  "eps": args.eps, "seed": args.seed},
        "final": {"tv": tv, "l1": l1, "logZ_learned": float(pol.logZ[0]),
                  "regression_slope": float(slope), "regression_r2": r2,
                  "spearman": sp, "mean_reward_ratio": mrr,
                  "modes_found": modes_found, "modes_target": n_modes},
        "curve": curve,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {args.out}")


if __name__ == "__main__":
    main()
