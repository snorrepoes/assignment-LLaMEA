import os
import textwrap
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from ioh import LogInfo, get_problem
from ioh.iohcpp.logger import AbstractLogger
from ioh.iohcpp.logger.trigger import ALWAYS

from llamea import ExperimentLogger, LLaMEA, Solution, Gemini_LLM
from llamea.utils import prepare_namespace

from google import genai
from google.genai import types


class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""


class AOCCLogger(AbstractLogger):
    """Computes the Area Over the Convergence Curve (AOCC) during a run."""

    def __init__(self, budget, lower=1e-8, upper=1e8, scale_log=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aocc = 0.0
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.transform = (lambda x: np.log10(x)) if scale_log else (lambda x: x)

    def __call__(self, log_info: LogInfo):
        if log_info.evaluations > self.budget:
            raise OverBudgetException
        if log_info.evaluations == self.budget:
            return
        y_value = np.clip(log_info.raw_y_best, self.lower, self.upper)
        self.aocc += (self.transform(y_value) - self.transform(self.lower)) / (
            self.transform(self.upper) - self.transform(self.lower)
        )

    def reset(self, func):
        super().reset()
        self.aocc = 0


def correct_aocc(ioh_function, logger, budget):
    """Normalizes the AOCC, correcting for runs that stopped early."""
    fraction = (
        logger.transform(np.clip(ioh_function.state.current_best_internal.y, logger.lower, logger.upper))
        - logger.transform(logger.lower)
    ) / (logger.transform(logger.upper) - logger.transform(logger.lower))
    aocc = (logger.aocc + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction) / budget
    return 1 - aocc


def evaluate(solution: Solution, explogger: Optional[ExperimentLogger] = None, **kwargs) -> Solution:
    """Runs a generated Optimize_with_BO class on a single BBOB function and scores it with the AOCC."""
    code_str = solution.code
    alg_name = solution.name

    try:
        ns, _ = prepare_namespace(
            code_str,
            allowed=["torch", "gpytorch", "botorch", "scipy", "numpy", "math"],
            logger=explogger,
        )
        exec(code_str, ns, ns)
        algorithm_class = ns[alg_name]
    except Exception as e:
        feedback = f"Error loading generated algorithm {alg_name}: {e}"
        solution.set_scores(-1.0, feedback)
        return solution

    dim = 2
    budget = 30 * dim
    logger = AOCCLogger(budget=budget, upper=1e2, triggers=[ALWAYS], scale_log=True)
    problem = get_problem(fid=16, instance=1, dimension=dim)
    problem.attach_logger(logger)

    try:
        algorithm = algorithm_class(budget=budget, dim=dim, initial_points=5 * dim, random_seed=1000)
        algorithm(problem)
    except OverBudgetException:
        pass
    except Exception as e:
        solution.set_scores(-1.0, f"Error running {alg_name}: {e}")
        return solution

    aocc = correct_aocc(problem, logger, budget)
    feedback = f"The algorithm {alg_name} achieved an AOCC of {aocc:.4f} (1.0 is best)."
    solution.set_scores(aocc, feedback)
    return solution


def main() -> None:
    load_dotenv()  # reads GEMINI_KEY from a local .env file

    api_key = os.environ.get("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_KEY is not set. Add it to your .env file.")

    # Pass your model name here (e.g., gemini-2.5-flash)
    llm = Gemini_LLM(api_key, "gemini-3.8-flash")
    
    role_prompt = "You are an expert in Bayesian optimization."
    task_prompt = textwrap.dedent("""\
        Create a Python class named Optimize_with_BO, with
        __init__(self, budget, dim, initial_points, random_seed) and a
        __call__(self, func) method that minimizes func over [-5, 5]^dim
        using BoTorch and GPyTorch, and returns the tuple (best_f, best_x).
        """)

    es = LLaMEA(
        f=evaluate,
        llm=llm,
        n_parents=1,
        n_offspring=1,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        experiment_name="minimal_example",
        elitism=False,
        minimization=False,
        budget=1,  # a single outer-loop generation -> a single LLM call
        log=False,
    )

    best_solution = es.run()
    print(f"Best solution: {best_solution.name}, score={best_solution.fitness:.4f}")
    print(best_solution.feedback)


if __name__ == "__main__":
    main()