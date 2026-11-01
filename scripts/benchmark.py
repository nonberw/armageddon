"""Solver shoot-out on random reachable 6-DoF poses. Prints a markdown table, saves assets/convergence.png.

    python scripts/benchmark.py --n 500
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from armageddon import HOME, default_arm, forward, solve  # noqa: E402

CONFIGS = {
    "Jacobian transpose": dict(method="transpose"),
    "Pseudo-inverse": dict(method="pinv"),
    "DLS (Levenberg-Marquardt)": dict(method="dls"),
    "DLS + 10 restarts": dict(method="dls", restarts=10),
}


def targets(chain, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        pose = forward(chain, rng.uniform(chain.lower * 0.8, chain.upper * 0.8))
        if pose.pos[2] > 0.05:
            out.append(pose)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    chain = default_arm()
    poses = targets(chain, args.n, args.seed)

    print(f"{args.n} random reachable poses (position + orientation), start from HOME, tol 0.1 mm / 0.06 deg\n")
    print("| Solver | Success | Median iters | Mean time / solve |")
    print("|---|---|---|---|")
    for name, cfg in CONFIGS.items():
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        res = [solve(chain, p.pos, p.rot, q0=HOME, rng=rng, **cfg) for p in poses]
        ms = (time.perf_counter() - t0) * 1e3 / len(poses)
        ok = [r for r in res if r.success]
        med = f"{np.median([r.iterations for r in ok]):.0f}" if ok else "-"
        print(f"| {name} | {100 * len(ok) / len(res):.1f}% | {med} | {ms:.2f} ms |")

    # Convergence plot: the farthest-from-HOME target that plain DLS still nails.
    home = forward(chain, HOME).pos
    solvable = [p for p in poses[:100] if solve(chain, p.pos, p.rot, q0=HOME).success]
    hard = max(solvable, key=lambda p: np.linalg.norm(p.pos - home))
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    ax.set_facecolor("#0d0b1a")
    colors = ["#ff5fa2", "#ffd166", "#2de2e6", "#9b5cff"]
    for (name, cfg), c in zip(list(CONFIGS.items())[:3], colors):
        r = solve(chain, hard.pos, hard.rot, q0=HOME, **cfg)
        ax.semilogy(r.history, label=f"{name} ({'ok' if r.success else 'fail'})", color=c, lw=2)
    ax.axhline(1e-4, color="#888", ls="--", lw=1)
    ax.text(1, 1.3e-4, "tolerance", color="#aaa", fontsize=9)
    ax.set_xlabel("iteration")
    ax.set_ylabel("pose error (weighted)")
    ax.set_title("Who gets there first?")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = ROOT / "assets" / "convergence.png"
    fig.savefig(out, facecolor="#0d0b1a")
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
