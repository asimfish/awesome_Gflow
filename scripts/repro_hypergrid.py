#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""??????????????? HyperGrid ??? GFlowNet???????

1. ???????T07/T01??log p_T(x) ? log R(x) ??????? < 1?
2. ?????T32??Spearman ????????????????????
   ?????? TV ??????????

??? numpy —— ?????? MLP?????????CPU ?????
????????????????????????????

???
    python3 scripts/repro_hypergrid.py                # ?? H=8, D=2
    python3 scripts/repro_hypergrid.py --H 12 --D 2   # ????
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# ???D ????? H ???????????????? +1???????
# ???? x ?????????? T01 ???????????????????
# --------------------------------------------------------------------------


def make_reward(H: int, D: int, R0: float = 1e-3, R1: float = 0.5, R2: float = 2.0):
    """T01 (Bengio et al. 2021) Eq. ????R0 + R1·[??] + R2·[??]?

    R0 ????????????????? —— ????????????
    """
    coords = np.array(list(itertools.product(range(H), repeat=D)), dtype=np.float64)
    z = np.abs(coords / (H - 1) - 0.5)
    shell1 = np.all(z > 0.25, axis=1)
    shell2 = np.all((z > 0.3) & (z < 0.4), axis=1)
    R = R0 + R1 * shell1 + R2 * shell2
    return coords.astype(int), R


def state_index(s, H: int, D: int) -> int:
    idx = 0
    for v in s:
        idx = idx * H + int(v)
    return idx


# --------------------------------------------------------------------------
# ???????? MLP??? one-hot ???????? D+1 ? logit
# ?? D ????? i ? +1????????????????????
# --------------------------------------------------------------------------


class Policy:
    def __init__(self, H: int, D: int, hidden: int = 128, seed: int = 0):
        self.H, self.D, self.hidden = H, D, hidden
        rng = np.random.default_rng(seed)
        din, dout = H * D, D + 1
        # He ???
        self.W1 = rng.normal(0, math.sqrt(2.0 / din), (din, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, math.sqrt(2.0 / hidden), (hidden, dout))
        self.b2 = np.zeros(dout)
        # log Z ? TB ?????????
        self.logZ = np.zeros(1)
        self._cache = None

    def encode(self, states: np.ndarray) -> np.ndarray:
        """states: (B, D) ???? -> (B, H*D) one-hot"""
        B = states.shape[0]
        x = np.zeros((B, self.H * self.D))
        for d in range(self.D):
            x[np.arange(B), d * self.H + states[:, d]] = 1.0
        return x

    def forward(self, states: np.ndarray):
        x = self.encode(states)
        h_pre = x @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)                      # ReLU
        logits = h @ self.W2 + self.b2
        self._cache = (x, h_pre, h)
        return logits

    def log_softmax(self, logits, mask):
        """mask: True ???????????????? -inf"""
        z = np.where(mask, -1e30, logits)
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    def backward(self, dlogits: np.ndarray, lr: float):
        x, h_pre, h = self._cache
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dh_pre = dh * (h_pre > 0)
        dW1 = x.T @ dh_pre
        db1 = dh_pre.sum(axis=0)
        for p, g in ((self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2)):
            np.clip(g, -10, 10, out=g)
            p -= lr * g


# --------------------------------------------------------------------------
# ??? TB ??
# --------------------------------------------------------------------------


def sample_batch(pol: Policy, batch: int, rng, eps: float = 0.0):
    """???????????????? (???, ????? sum log P_F, ????)?

    ????????????? s ??????? = s ???????
    ?? log P_B(?) = -?_t log(#parents(s_t))???????
    """
    H, D = pol.H, pol.D
    s = np.zeros((batch, D), dtype=int)
    alive = np.ones(batch, dtype=bool)
    logpf = np.zeros(batch)
    logpb = np.zeros(batch)
    steps = np.zeros(batch, dtype=int)
    # ?????????????????
    tape = []

    for _ in range(D * (H - 1) + 1):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        cur = s[idx]
        logits = pol.forward(cur)
        at_edge = cur >= H - 1                                    # (n, D)
        mask = np.concatenate([at_edge, np.zeros((len(idx), 1), bool)], axis=1)
        logp = pol.log_softmax(logits, mask)
        p = np.exp(logp)
        if eps > 0:                                              # ?-greedy ??
            legal = ~mask
            u = legal / legal.sum(axis=1, keepdims=True)
            p = (1 - eps) * p + eps * u
            p /= p.sum(axis=1, keepdims=True)
        # ????
        c = p.cumsum(axis=1)
        r = rng.random((len(idx), 1))
        act = (r > c).sum(axis=1)
        act = np.minimum(act, D)                                  # ????
        logpf[idx] += logp[np.arange(len(idx)), act]
        tape.append((cur.copy(), mask.copy(), act.copy(), idx.copy()))

        stop = act == D
        move = ~stop
        if move.any():
            mi = idx[move]
            s[mi, act[move]] += 1
            steps[mi] += 1
            # ??????????? = ?????
            nparents = np.maximum((s[mi] > 0).sum(axis=1), 1)
            logpb[mi] -= np.log(nparents)
        alive[idx[stop]] = False

    return s, logpf, logpb, steps, tape


def tb_step(pol: Policy, batch: int, rng, R_table, H, D, lr: float, eps: float):
    """?? Trajectory Balance ???

    TB loss = ( logZ + log P_F(?) - log R(x) - log P_B(?|x) )^2
    """
    s, logpf, logpb, steps, tape = sample_batch(pol, batch, rng, eps)
    flat = np.array([state_index(x, H, D) for x in s])
    logR = np.log(R_table[flat])
    delta = pol.logZ[0] + logpf - logR - logpb
    loss = (delta ** 2).mean()

    # d loss / d logZ
    pol.logZ -= lr * np.clip(2.0 * delta.mean(), -10, 10)

    # d loss / d logP_F(?) = 2·delta/B?????? log_softmax ??
    coef = 2.0 * delta / batch
    for cur, mask, act, idx in reversed(tape):
        logits = pol.forward(cur)
        logp = pol.log_softmax(logits, mask)
        p = np.exp(logp)
        # d(logp[act]) / d logits = onehot(act) - p
        g = -p.copy()
        g[np.arange(len(idx)), act] += 1.0
        dlogits = -(coef[idx][:, None] * g)     # ?????? loss ? ????
        dlogits[mask] = 0.0
        pol.backward(dlogits, lr)
    return loss, steps.mean()


# --------------------------------------------------------------------------
# ???????????
# --------------------------------------------------------------------------


def empirical_terminal_dist(pol: Policy, n: int, rng, H, D, chunk: int = 2000):
    """????????? P_T(x)?HyperGrid ??????????????"""
    counts = np.zeros(H ** D)
    got = 0
    while got < n:
        b = min(chunk, n - got)
        s, _, _, _, _ = sample_batch(pol, b, rng, eps=0.0)
        for x in s:
            counts[state_index(x, H, D)] += 1
        got += b
    return counts / counts.sum()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--eval-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="artifacts/repro_hypergrid.json")
    args = ap.parse_args()

    H, D = args.H, args.D
    rng = np.random.default_rng(args.seed)
    coords, R = make_reward(H, D)
    Z_true = R.sum()
    P_star = R / Z_true

    pol = Policy(H, D, seed=args.seed)
    print(f"??: {D} ? HyperGrid??? {H}?? {H**D} ????")
    print(f"?? Z = {Z_true:.4f}????? {R.max():.3f}??? {R.min():.4f}")
    print(f"?????????R > 1.0 ????: {(R > 1.0).sum()}\n")

    log = []
    for it in range(1, args.iters + 1):
        loss, meanlen = tb_step(pol, args.batch, rng, R, H, D, args.lr, args.eps)
        if it % max(1, args.iters // 10) == 0 or it == 1:
            Pt = empirical_terminal_dist(pol, 4000, rng, H, D)
            tv = 0.5 * np.abs(Pt - P_star).sum()
            log.append({"iter": it, "loss": float(loss), "tv": float(tv),
                        "logZ": float(pol.logZ[0]), "mean_len": float(meanlen)})
            print(f"iter {it:>5} | TB loss {loss:>9.4f} | TV {tv:.4f} | "
                  f"logZ {pol.logZ[0]:>7.3f} (?? {math.log(Z_true):.3f}) | ??? {meanlen:.2f}")

    # ---------------- ???? ----------------
    print("\n" + "=" * 74)
    Pt = empirical_terminal_dist(pol, args.eval_samples, rng, H, D)
    tv = 0.5 * np.abs(Pt - P_star).sum()
    l1 = np.abs(Pt - P_star).sum()

    # ?? 1??????? —— log p ? log R ?????
    m = Pt > 0
    lp, lr_ = np.log(Pt[m]), np.log(R[m])
    A = np.vstack([lr_, np.ones_like(lr_)]).T
    slope, intercept = np.linalg.lstsq(A, lp, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_res = ((lp - pred) ** 2).sum()
    ss_tot = ((lp - lp.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # ?? 2????? —— ?????????????? TV
    sp = spearman(Pt, R)
    mean_reward_ratio = float((Pt * R).sum() / (P_star * R).sum())
    modes_target = int((R > 1.0).sum())
    modes_found = int(((R > 1.0) & (Pt > 1.0 / (H ** D) / 10)).sum())

    result = {
        "env": {"H": H, "D": D, "n_terminal": H ** D, "Z_true": float(Z_true),
                "modes_target": modes_target},
        "train": {"iters": args.iters, "batch": args.batch, "lr": args.lr, "eps": args.eps},
        "final": {
            "tv": float(tv), "l1": float(l1),
            "logZ_learned": float(pol.logZ[0]), "logZ_true": float(math.log(Z_true)),
            "regression_slope": float(slope), "regression_r2": float(r2),
            "spearman_Pt_vs_R": float(sp),
            "mean_reward_ratio": mean_reward_ratio,
            "modes_found": modes_found,
        },
        "curve": log,
    }

    print("??? 1????????T07 / T01?")
    print(f"  log P_T(x) ? log R(x) ????? = {slope:.3f}??? 1.0??R² = {r2:.3f}")
    print(f"  ???? T01 ???????????? 0.58 —— ?? < 1 ??"
          f"{'????????????????' if slope < 0.95 else '?????'}")
    print(f"  ??? logZ = {pol.logZ[0]:.3f}??? log Z = {math.log(Z_true):.3f}?"
          f"?? {abs(pol.logZ[0]-math.log(Z_true)):.3f}")

    print("\n??? 2???????????????T32?")
    print(f"  ??????:      TV = {tv:.4f}   L1 = {l1:.4f}   ? ???????")
    print(f"  Spearman ??:     {sp:+.4f}      ? ?????????????")
    print(f"  ?????:        {mean_reward_ratio:.4f}      ? ?? 1 ???????")
    print(f"  ?????:        {modes_found}/{modes_target}          ? ???????")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n????? {args.out}")


if __name__ == "__main__":
    main()
