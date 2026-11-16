"""Area Over the Convergence Curve on log-scaled regret (higher = better, in [0,1]).
AOCC after the LLaMEA-BO paper (Li, van Stein, Baeck, Raponi, GECCO'26); regret bounds are our own choice."""
import numpy as np


def best_so_far_regret(values, f_opt, budget):
    r = np.maximum(np.asarray(values, float) - f_opt, 0.0)
    curve = np.minimum.accumulate(r) if r.size else np.array([])
    return curve  # length <= budget; missing evaluations are treated as worst in aocc()


def aocc(values, f_opt, budget, lb, ub):
    curve = best_so_far_regret(values, f_opt, budget)
    full = np.full(budget, ub, float)         # unused evaluations count as worst (ub)
    full[: min(budget, curve.size)] = curve[:budget]
    full = np.clip(full, lb, ub)
    s = (np.log10(full) - np.log10(lb)) / (np.log10(ub) - np.log10(lb))
    return float(1.0 - s.mean())
