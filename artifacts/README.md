# artifacts/

复现实验的原始输出。由 `scripts/repro_hypergrid.py` 生成，可用同一命令重跑复核。

| 文件 | 生成命令 | 用途 |
|---|---|---|
| `repro_hypergrid.json` | `python3 scripts/repro_hypergrid.py --H 8 --D 2 --iters 4000` | 欠训练组（TV 0.454） |
| `repro_hypergrid_long.json` | `python3 scripts/repro_hypergrid.py --H 8 --D 2 --iters 15000 --lr 0.01 --out artifacts/repro_hypergrid_long.json` | 收敛组（TV 0.015） |

两组构成对照，用来实证「流行评测指标不反映分布正确性」——详见 [insights/repro_verification.md](../insights/repro_verification.md)。
