"""Problem factory + budget-enforcing wrapper. All problems are minimisation in [-5,5]^d."""
import json, pathlib
import numpy as np

CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "experiment.json"


def load_config(path=CFG_PATH):
    with open(path) as fh:
        return json.load(fh)


class BudgetExhausted(Exception):
    pass


class _Bounds:
    def __init__(self, lb, ub):
        self.lb, self.ub = np.asarray(lb, float), np.asarray(ub, float)


class Recorder:
    """Wraps a raw problem: counts every call (initial design included), enforces the
    budget and stores raw values. Exposes func.bounds.lb/ub, func.dim, func(x)."""

    def __init__(self, problem, budget, dim):
        self.problem, self.budget, self.dim = problem, int(budget), int(dim)
        self.bounds = _Bounds(-5.0 * np.ones(dim), 5.0 * np.ones(dim))
        self.values = []

    def __call__(self, x):
        if len(self.values) >= self.budget:
            raise BudgetExhausted()
        x = np.clip(np.asarray(x, float).reshape(-1), -5.0, 5.0)
        if x.size != self.dim:
            raise ValueError(f"expected a single point of dimension {self.dim}, got {x.size}")
        y = float(self.problem(x.tolist()))
        self.values.append(y)
        return y


def make_raw_problem(target, cfg=None):
    """Returns (ioh-style problem object, f_opt) for a target key."""
    cfg = cfg or load_config()
    t, d = cfg["targets"][target], cfg["dim"]
    if t["kind"] == "bbob":
        import ioh
        p = ioh.get_problem(t["fid"], instance=t["instance"], dimension=d)
        return p, float(p.optimum.y)
    from Practical_Problems import get_practical_problem
    return get_practical_problem(t["pid"], dim=d), float(t["f_opt"])


def make_recorder(target, budget, cfg=None):
    cfg = cfg or load_config()
    p, f_opt = make_raw_problem(target, cfg)
    return Recorder(p, budget, cfg["dim"]), f_opt
