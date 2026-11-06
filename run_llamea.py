"""Run one LLaMEA search for a target.  usage: python run_llamea.py bbob_f21|lj1125 [--run-tag tag]"""
import argparse, os, sys, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text().splitlines():          # simple .env loader (no extra dependency)
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("OMP_NUM_THREADS", "1")

from llamea import LLaMEA
from boa.llm import RobustGemini, RobustCerebras, RobustMistral
from boa.problems import load_config
from boa.evaluator import Evaluator

ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--run-tag", default="run1")
ap.add_argument("--budget", type=int, default=None); ap.add_argument("--llm", choices=["gemini", "cerebras", "mistral"], default="gemini")
ap.add_argument("--workers", type=int, default=None); ap.add_argument("--model", default=None); a = ap.parse_args()
cfg = load_config(); lc = cfg["llamea"]
task = (ROOT / "prompts/base_task.txt").read_text() + "\n" + (ROOT / f"prompts/hint_{a.target}.txt").read_text()
if a.llm == "gemini":
    llm = RobustGemini(os.environ["GOOGLE_API_KEY"], model=lc["model"]); provider, model = "google_ai_studio", lc["model"]
elif a.llm == "mistral":
    model = a.model or lc.get("mistral_model", "codestral-latest"); llm = RobustMistral(os.environ["MISTRAL_API_KEY"], model=model, reasoning_effort=lc.get("mistral_reasoning_effort")); provider = "mistral"
else:
    model = lc.get("cerebras_model", "gpt-oss-120b"); llm = RobustCerebras(os.environ["CEREBRAS_API_KEY"], model=model, reasoning_effort=lc.get("cerebras_reasoning_effort", "low")); provider = "cerebras"
import json, time
(ROOT / "results" / "llamea_logs").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "llamea_logs" / f"{a.target}_{a.run_tag}.meta.json").write_text(json.dumps(
    {"target": a.target, "provider": provider, "model": model, "run_tag": a.run_tag, "started": time.strftime("%F %T"),
     "llamea_cfg": lc, "workers": a.workers or lc["max_workers"], "seeds": cfg["seeds"]}, indent=2))
exp = LLaMEA(f=Evaluator(a.target), llm=llm, budget=a.budget or lc["outer_budget"], n_parents=lc["n_parents"],
             n_offspring=lc["n_offspring"], elitism=lc["elitism"], role_prompt="You are a highly skilled computer scientist in the field of Bayesian optimization and Python programming.",
             task_prompt=task, eval_timeout=lc["eval_timeout_s"], max_workers=a.workers or lc["max_workers"],
             experiment_name=f"{a.target}_{a.run_tag}", log=True, minimization=False)
best = exp.run()
print("BEST", best.name, best.fitness)
(ROOT / "results" / f"best_{a.target}_{a.run_tag}.py").write_text(best.code)
