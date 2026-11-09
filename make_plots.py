"""Convergence plots (mean +- s.e. over 10 seeds) for stage 2 + LLaMEA search curves; summary table."""
import json, glob, pathlib, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from boa.problems import load_config, make_raw_problem
cfg = load_config(); S2 = ROOT / "results" / "stage2"; FIG = ROOT / "results" / "figures"; FIG.mkdir(exist_ok=True, parents=True)
NAMES = {"cmaes": "CMA-ES", "botorch_bo": "BoTorch BO", "llamea_bbob_f21": "LLaMEA-BO (gen. on f21)", "llamea_lj1125": "LLaMEA-BO (gen. on LJ)"}
TITLE = {"bbob_f21": "BBOB f21 (instance 1), d=6", "lj1125": "Lennard-Jones (1125), d=6"}
rows = []
for prob in ["bbob_f21", "lj1125"]:
    _, f_opt = make_raw_problem(prob, cfg); fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, nm in NAMES.items():
        runs = [np.load(f) for f in sorted(glob.glob(str(S2 / prob / f"{m}_seed*.npy")))]
        if not runs: continue
        B = max(len(r) for r in runs)
        C = np.array([np.minimum.accumulate(np.maximum(np.pad(r, (0, B - len(r)), mode="edge") - f_opt, 1e-12)) for r in runs])
        mu, se = C.mean(0), C.std(0, ddof=1) / np.sqrt(len(C)) if len(C) > 1 else np.zeros(B)
        x = np.arange(1, B + 1); ax.plot(x, mu, label=f"{nm} (n={len(C)})"); ax.fill_between(x, np.maximum(mu - se, 1e-12), mu + se, alpha=.2)
        rows.append((prob, nm, len(C), mu[-1], se[-1]))
    ax.set_yscale("log"); ax.set_xlabel("function evaluations"); ax.set_ylabel("best-so-far regret f - f*")
    ax.set_title(TITLE[prob]); ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / f"convergence_{prob}.png", dpi=150); plt.close(fig)
(FIG / "summary.md").write_text("| problem | method | runs | final regret (mean) | s.e. |\n|---|---|---|---|---|\n" +
    "\n".join(f"| {p} | {n} | {k} | {m:.4g} | {s:.2g} |" for p, n, k, m, s in rows))
# LLaMEA search curves
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, prob in zip(axs, ["bbob_f21", "lj1125"]):
    for d in sorted(glob.glob(str(ROOT / "results/llamea_logs" / f"exp-*{prob}_run2"))):
        fit = [float(json.loads(l)["fitness"]) for l in open(d + "/log.jsonl")]
        fit = np.array([v if np.isfinite(v) else np.nan for v in fit]); x = np.arange(1, len(fit) + 1)
        ax.scatter(x, fit, s=8, alpha=.5, label="candidate AOCC"); ax.plot(x, np.fmax.accumulate(np.nan_to_num(fit, nan=-1)), "r", label="best so far")
    ax.set_title(f"LLaMEA search: {TITLE[prob]}"); ax.set_xlabel("candidate #"); ax.set_ylabel("AOCC"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "llamea_search_convergence.png", dpi=150)
print((FIG / "summary.md").read_text())
