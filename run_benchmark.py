"""Stage 2: every method x problem x seed, budget 100d=600. Resumable; one .npy of raw f-values per run.
usage: python run_benchmark.py [--gen bbob_f21=path.py --gen lj1125=path.py] [--methods ...] [--procs 6] [--budget 600]"""
import argparse, os, sys, pathlib, warnings, importlib.util, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from boa.problems import load_config, make_recorder, BudgetExhausted
from boa.initial_design import initial_points
OUT = ROOT / "results" / "stage2"
PROBLEMS = ["bbob_f21", "lj1125"]


def load_class(method, gen_paths):
    if method == "botorch_bo":
        from baselines.botorch_bo import Optimize_with_BO as C; return C
    if method == "cmaes":
        from baselines.cmaes import Optimize_with_CMAES as C; return C
    if method.startswith("llamea_"):                       # llamea_<generation target>
        path = gen_paths[method[len("llamea_"):]]
        ns = {"__name__": f"gen_{method}"}
        exec(compile(pathlib.Path(path).read_text(), path, "exec"), ns)
        return ns["Optimize_with_BO"]
    raise ValueError(method)


def job(args):
    method, problem, seed, budget, gen_paths = args
    f = OUT / problem / f"{method}_seed{seed}.npy"
    if f.exists(): return str(f), "cached"
    import torch; torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    cfg = load_config(); rec, _ = make_recorder(problem, budget, cfg)
    algo = load_class(method, gen_paths)(budget=budget, dim=cfg["dim"], random_seed=seed,
                                         initial_points=initial_points(cfg["n_init"], cfg["dim"], seed))
    try: algo(rec)
    except BudgetExhausted: pass
    except Exception as e:                                     # keep partial trace, record the failure
        (f.parent / f"{method}_seed{seed}.error.txt").write_text(repr(e))
    f.parent.mkdir(parents=True, exist_ok=True); np.save(f, np.array(rec.values))
    return str(f), f"{len(rec.values)} evals"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append", default=[]); ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--methods", nargs="*", default=["cmaes", "botorch_bo", "llamea_bbob_f21", "llamea_lj1125"])
    ap.add_argument("--budget", type=int, default=None); ap.add_argument("--problems", nargs="*", default=PROBLEMS)
    a = ap.parse_args(); cfg = load_config()
    gen = dict(g.split("=", 1) for g in a.gen); budget = a.budget or cfg["stage2_budget_evals"]
    for p in a.problems: (OUT / p).mkdir(parents=True, exist_ok=True)
    jobs = [(m, p, s, budget, gen) for m, p, s in itertools.product(a.methods, a.problems, cfg["seeds"])]
    with ProcessPoolExecutor(a.procs) as ex:
        for path, st in ex.map(job, jobs): print(path, st, flush=True)
