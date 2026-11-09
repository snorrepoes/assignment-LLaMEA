import random
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        self.random_seed = int(random_seed)
        self._seed_everything()

    def _seed_everything(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        torch.set_default_dtype(torch.float64)

    def __call__(self, func):
        self._seed_everything()

        lb = np.asarray(func.bounds.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(func.bounds.ub, dtype=np.float64).reshape(-1)
        bounds = torch.tensor(np.vstack([lb, ub]), dtype=torch.float64)

        X = []
        Y = []

        def evaluate(x):
            x = np.clip(np.asarray(x, dtype=np.float64).reshape(-1), lb, ub)
            y = float(func(x))
            X.append(x.copy())
            Y.append(y)
            return y

        # Evaluate the provided initial design first.
        for x0 in self.initial_points:
            if len(X) >= self.budget:
                break
            evaluate(x0)

        best_idx = int(np.argmin(Y))
        best_f = Y[best_idx]
        best_x = X[best_idx].copy()

        # Bayesian optimization loop, respecting the total evaluation budget.
        while len(X) < self.budget:
            train_X = torch.tensor(np.asarray(X), dtype=torch.float64)
            train_Y = torch.tensor(np.asarray(Y), dtype=torch.float64).unsqueeze(-1)

            gp = SingleTaskGP(
                train_X,
                train_Y,
                covar_module=ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=self.dim)
                ),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            acq = qLogExpectedImprovement(model=gp, best_f=best_f)

            try:
                candidate, _ = optimize_acqf(
                    acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=20,
                    raw_samples=256,
                    options={"maxiter": 200},
                )
                x_next = candidate.detach().cpu().numpy().reshape(-1)
            except Exception:
                x_next = np.random.uniform(lb, ub)

            x_next = np.clip(x_next, lb, ub)
            y_next = evaluate(x_next)

            if y_next < best_f:
                best_f = y_next
                best_x = x_next.copy()

        return best_f, best_x