"""LLaMEA fitness function: runs a candidate Optimize_with_BO on one target for all seeds.
Follows the LLaMEA fitness-function pattern (https://github.com/XAI-liacs/LLaMEA) and uses llamea.utils; AOCC after the LLaMEA-BO paper (Li et al., GECCO'26)."""
import os, signal, traceback
import numpy as np
from boa.problems import load_config, make_recorder, BudgetExhausted
from boa.aocc import aocc
from boa.initial_design import initial_points

ALLOWED = ["numpy", "scipy", "torch", "botorch", "gpytorch", "math", "random", "warnings", "typing", "time"]
CLASS_NAME = "Optimize_with_BO"


def run_algorithm(cls, target, seed, budget, cfg):
    """One independent run; returns raw function values (<= budget) of the run."""
    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    rec, f_opt = make_recorder(target, budget, cfg)
    algo = cls(budget=budget, dim=cfg["dim"], initial_points=initial_points(cfg["n_init"], cfg["dim"], seed),
               random_seed=seed)
    try:
        algo(rec)
    except BudgetExhausted:
        pass
    return rec.values, f_opt


class CandidateTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CandidateTimeout()


class Evaluator:
    """Picklable callable: f(solution, logger) -> solution (LLaMEA API)."""

    def __init__(self, target, config_path=None):
        self.target, self.config_path = target, config_path

    def __call__(self, solution, logger=None):
        from llamea.utils import prepare_namespace, clean_local_namespace
        import torch; torch.set_num_threads(1)
        cfg = load_config(self.config_path) if self.config_path else load_config()
        lb, ub = cfg["targets"][self.target]["aocc_regret_bounds"]
        budget = cfg["stage1_budget_evals"]
        limit = int(cfg["llamea"]["candidate_time_limit_s"])
        try:                                   # own time limit: a slow candidate is a failure, not a crash
            signal.signal(signal.SIGALRM, _alarm); signal.alarm(limit)
        except ValueError:                     # not in main thread: rely on LLaMEA's backstop timeout
            pass
        try:
            gns, _ = prepare_namespace(solution.code, ALLOWED, logger)
            lns = {}
            exec(solution.code, gns, lns)
            clean_local_namespace(lns, gns)
            cls = lns.get(CLASS_NAME) or gns.get(CLASS_NAME)
            scores, finals = [], []
            for seed in cfg["seeds"]:
                vals, f_opt = run_algorithm(cls, self.target, seed, budget, cfg)
                if len(vals) == 0:
                    raise RuntimeError("algorithm made no function evaluations")
                scores.append(aocc(vals, f_opt, budget, lb, ub)); finals.append(min(vals) - f_opt)
            m, s = float(np.mean(scores)), float(np.std(scores))
            fb = (f"The algorithm {solution.name} got a mean AOCC of {m:.4f} (std {s:.4f}) over {len(scores)} runs "
                  f"(higher is better, max 1). Final best regret per run: "
                  + ", ".join(f"{v:.3g}" for v in finals) + ". Per-run AOCC: " + ", ".join(f"{v:.3f}" for v in scores) + ".")
            solution.set_scores(m, fb)
        except CandidateTimeout:
            solution.set_scores(-np.inf, f"The algorithm exceeded the time limit of {limit}s for 10 runs of {budget} "
                                "evaluations. Make it much faster (fewer restarts/raw samples, cheaper acquisition "
                                "optimisation, avoid refitting large ensembles every iteration).")
        except Exception as e:  # noqa
            solution.set_scores(-np.inf, f"{type(e).__name__}: {e}", error=e)
        finally:
            try: signal.alarm(0)
            except Exception: pass
        return solution
