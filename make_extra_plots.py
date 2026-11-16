"""Extra figures for the README: final-regret distributions, per-seed traces, success rate vs evaluations,
early-budget zoom, cross-problem heatmap, LLaMEA failure causes and AOCC per generation.
Same data and colours as make_plots.py. usage: python make_extra_plots.py"""
import json, glob, re, pathlib, sys, collections
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from boa.problems import load_config, make_raw_problem
cfg = load_config(); S2 = ROOT / "results" / "stage2"; FIG = ROOT / "results" / "figures"
NAMES = {"cmaes": "CMA-ES", "botorch_bo": "BoTorch BO", "llamea_bbob_f21": "LLaMEA-BO (gen. on f21)", "llamea_lj1125": "LLaMEA-BO (gen. on LJ)"}
SHORT = {"cmaes": "CMA-ES", "botorch_bo": "BoTorch BO", "llamea_bbob_f21": "gen. on f21", "llamea_lj1125": "gen. on LJ"}
COL = dict(zip(NAMES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))       # same colours as the convergence plots
MARK = dict(zip(NAMES, ["o", "s", "^", "D"]))                               # second encoding besides colour
TITLE = {"bbob_f21": "BBOB f21 (instance 1), d=6", "lj1125": "Lennard-Jones (1125), d=6"}
PROBS = ["bbob_f21", "lj1125"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})

def curves(prob, m, f_opt):
    runs = [np.load(f) for f in sorted(glob.glob(str(S2 / prob / f"{m}_seed*.npy")))]
    B = max(len(r) for r in runs)
    return np.array([np.minimum.accumulate(np.maximum(np.pad(r, (0, B - len(r)), mode="edge") - f_opt, 1e-12)) for r in runs])

C = {}
for p in PROBS:
    _, f_opt = make_raw_problem(p, cfg)
    for m in NAMES: C[p, m] = curves(p, m, f_opt)

# 1. final regret distribution (all 10 seeds as dots, median bar) ---------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, p in zip(axs, PROBS):
    for i, m in enumerate(NAMES):
        v = C[p, m][:, -1]; rng = np.random.default_rng(0); xs = i + rng.uniform(-.14, .14, len(v))
        ax.scatter(xs, v, s=28, c=COL[m], marker=MARK[m], alpha=.8, edgecolors="white", linewidths=.5, zorder=3)
        ax.hlines(np.median(v), i - .3, i + .3, colors="black", lw=2, zorder=4)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[m] for m in NAMES], fontsize=8)
    ax.set_title(TITLE[p]); ax.set_ylabel("final regret after 600 evaluations")
fig.suptitle("Final regret per seed (dots) and median (black bar)", fontsize=10); fig.tight_layout(); fig.savefig(FIG / "final_regret_distribution.png", dpi=150); plt.close(fig)

# 2. every seed's trace (small multiples) -------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey="row")
for r, p in enumerate(PROBS):
    for c, m in enumerate(NAMES):
        ax = axs[r, c]; X = C[p, m]; x = np.arange(1, X.shape[1] + 1)
        for row in X: ax.plot(x, row, color=COL[m], alpha=.35, lw=1)
        ax.plot(x, np.median(X, 0), color="black", lw=2); ax.set_yscale("log")
        if r == 0: ax.set_title(SHORT[m], fontsize=10)
        if c == 0: ax.set_ylabel(("f21" if p == "bbob_f21" else "LJ") + "\nbest-so-far regret")
        if r == 1: ax.set_xlabel("evaluations")
fig.suptitle("Individual runs (coloured, 10 seeds) and median (black)", fontsize=10); fig.tight_layout(); fig.savefig(FIG / "individual_runs.png", dpi=150); plt.close(fig)

# 3. success rate vs evaluations --------------------------------------------------------------------
TH = {"bbob_f21": 5.0, "lj1125": 1e-2}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
for ax, p in zip(axs, PROBS):
    for m in NAMES:
        X = C[p, m]; x = np.arange(1, X.shape[1] + 1)
        ax.step(x, (X <= TH[p]).mean(0), where="post", color=COL[m], lw=2, label=NAMES[m])
    ax.set_ylim(-.03, 1.03); ax.set_xlabel("function evaluations"); ax.set_ylabel("fraction of the 10 runs")
    ax.set_title(f"{TITLE[p]}\nbest-so-far regret ≤ {TH[p]:g}", fontsize=10)
axs[0].legend(fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(FIG / "success_rate.png", dpi=150); plt.close(fig)

# 4. early-budget zoom (selection used 60 evaluations) ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
for ax, p in zip(axs, PROBS):
    for m in NAMES:
        X = C[p, m][:, :100]; x = np.arange(1, 101); mu = X.mean(0); se = X.std(0, ddof=1) / np.sqrt(len(X))
        ax.plot(x, mu, color=COL[m], lw=2, label=NAMES[m]); ax.fill_between(x, np.maximum(mu - se, 1e-12), mu + se, color=COL[m], alpha=.18)
    ax.axvline(60, color="gray", ls="--", lw=1); ax.text(61, ax.get_ylim()[1] * .6, "Stage-1\nbudget (60)", fontsize=7, color="gray", va="top")
    ax.set_yscale("log"); ax.set_xlabel("function evaluations"); ax.set_ylabel("best-so-far regret"); ax.set_title(TITLE[p], fontsize=10)
axs[0].legend(fontsize=7, loc="lower left"); fig.tight_layout(); fig.savefig(FIG / "early_budget_zoom.png", dpi=150); plt.close(fig)

# 5. cross-problem heatmap -------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
M = np.array([[np.log10(C[p, m][:, -1].mean()) for p in PROBS] for m in NAMES]); Z = M - M.min(0)   # colour: within-problem, 0 = best
im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=Z.max())
for i, m in enumerate(NAMES):
    for j, p in enumerate(PROBS):
        v = C[p, m][:, -1].mean(); own = (m == "llamea_bbob_f21" and p == "bbob_f21") or (m == "llamea_lj1125" and p == "lj1125")
        ax.text(j, i, f"{v:.3g}" + ("\n(own target)" if own else ""), ha="center", va="center", fontsize=8, color="white" if Z[i, j] > .6 * Z.max() else "black")
ax.set_xticks(range(2)); ax.set_xticklabels(["BBOB f21", "LJ 1125"]); ax.set_yticks(range(4)); ax.set_yticklabels([NAMES[m] for m in NAMES], fontsize=8)
ax.grid(False); ax.set_title("Mean final regret (darker = worse, within each problem)", fontsize=9); fig.tight_layout(); fig.savefig(FIG / "generalization_heatmap.png", dpi=150); plt.close(fig)

# 6/7. LLaMEA search: failure causes and AOCC per generation -----------------------------------------------------------
def cause(r):
    if r["fitness"] is not None and np.isfinite(float(r["fitness"])): return None
    if "time limit" in (r.get("feedback") or ""): return "time limit (1200 s)"
    e = r.get("error") or ""
    if "ModuleNotFoundError" in e or "ImportError" in e: return "import of a non-existent BoTorch/GPyTorch module"
    if "ModelFittingError" in e: return "GP fitting failed"
    if "AttributeError" in e: return "non-existent attribute / method"
    if "TypeError" in e: return "wrong function arguments"
    if "RuntimeError" in e or "InputDataError" in e or "OptimizationGradientError" in e: return "runtime / tensor error"
    return "other"
logs = {p: [json.loads(l) for l in open(glob.glob(str(ROOT / "results/llamea_logs" / f"exp-*{p}_run2"))[0] + "/log.jsonl")] for p in PROBS}
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
for ax, p in zip(axs, PROBS):
    cnt = collections.Counter(c for c in map(cause, logs[p]) if c); lab = sorted(cnt, key=cnt.get)
    ax.barh(lab, [cnt[k] for k in lab], color="#777777", height=.6)
    for i, k in enumerate(lab): ax.text(cnt[k] + .3, i, str(cnt[k]), va="center", fontsize=8)
    ax.set_title(f"{TITLE[p]}\n{sum(cnt.values())} of {len(logs[p])} candidates failed", fontsize=9); ax.set_xlabel("candidates"); ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(FIG / "llamea_failure_causes.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, p in zip(axs, PROBS):
    g = collections.defaultdict(list)
    for r in logs[p]:
        if cause(r) is None: g[r["generation"]].append(float(r["fitness"]))
    ks = sorted(g)
    ax.boxplot([g[k] for k in ks], positions=ks, widths=.5, showfliers=False, medianprops=dict(color="black"))
    for k in ks: ax.scatter(np.full(len(g[k]), k) + np.random.default_rng(1).uniform(-.12, .12, len(g[k])), g[k], s=10, color="#1f77b4", alpha=.6, zorder=3)
    ax.set_xlabel("LLaMEA generation"); ax.set_ylabel("AOCC of valid candidates"); ax.set_title(TITLE[p], fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "llamea_aocc_by_generation.png", dpi=150); plt.close(fig)
print("written:", sorted(p.name for p in FIG.glob("*.png")))
