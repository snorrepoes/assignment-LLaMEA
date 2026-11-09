import random
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        if self.initial_points.ndim != 2 or self.initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must have shape (n_init, dim)")
        if self.initial_points.shape[0] == 0:
            raise ValueError("initial_points must not be empty")
        if self.initial_points.shape[0] > self.budget:
            raise ValueError("initial_points count exceeds budget")
        self.random_seed = int(random_seed)
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

    def _set_seed(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def _evaluate(self, func, x):
        if self._call_count >= self.budget:
            raise RuntimeError("Evaluation budget exceeded")
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        x = np.clip(np.asarray(x, dtype=np.float64).reshape(-1), lb, ub)
        if x.shape[0] != self.dim:
            raise ValueError("x has wrong dimension")
        f = float(func(x))
        self._call_count += 1
        if f < self._best_f:
            self._best_f = f
            self._best_x = x.copy()
        return f, x

    def __call__(self, func):
        self._set_seed()
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        self.bounds = torch.tensor(np.vstack([lb, ub]), dtype=torch.float64)

        X_obs = []
        Y_obs = []

        for x in self.initial_points:
            f, x_eval = self._evaluate(func, x)
            X_obs.append(x_eval)
            Y_obs.append(f)

        while self._call_count < self.budget:
            torch.manual_seed(self.random_seed + self._call_count)
            np.random.seed(self.random_seed + self._call_count)
            random.seed(self.random_seed + self._call_count)

            X_tensor = torch.tensor(np.array(X_obs), dtype=torch.float64)
            Y_tensor = torch.tensor(np.array(Y_obs), dtype=torch.float64).unsqueeze(-1)

            X_norm = (X_tensor - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

            model = SingleTaskGP(X_norm, Y_tensor)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_f = torch.tensor(self._best_f, dtype=torch.float64)
            acq = qLogExpectedImprovement(model, best_f=best_f)

            standard_bounds = torch.tensor(
                [[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64
            )

            torch.manual_seed(self.random_seed + self._call_count + 1000)
            candidate, _ = optimize_acqf(
                acq,
                bounds=standard_bounds,
                q=1,
                num_restarts=10,
                raw_samples=64,
                options={"maxiter": 200},
            )

            x_candidate = candidate[0].detach().cpu().numpy()
            x_orig = (
                self.bounds[0].cpu().numpy()
                + x_candidate * (self.bounds[1] - self.bounds[0]).cpu().numpy()
            )
            x_orig = np.clip(x_orig, lb, ub)

            f, x_eval = self._evaluate(func, x_orig)
            X_obs.append(x_eval)
            Y_obs.append(f)

        return self._best_f, self._best_x