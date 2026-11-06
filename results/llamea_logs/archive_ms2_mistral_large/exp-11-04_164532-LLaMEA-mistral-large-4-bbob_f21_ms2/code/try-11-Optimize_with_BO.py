import random
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.random_seed = int(random_seed)
        self.initial_points = np.asarray(initial_points, dtype=np.float64)

        if self.initial_points.ndim != 2 or self.initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must have shape (n_init, dim)")
        if self.initial_points.shape[0] > self.budget:
            raise ValueError("Number of initial points exceeds budget")

        torch.set_default_dtype(torch.float64)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self.n_calls = 0
        self.train_X = []
        self.train_Y = []
        self.best_f = np.inf
        self.best_x = None
        self.lb = None
        self.ub = None

    def _evaluate(self, func, x):
        if self.n_calls >= self.budget:
            raise RuntimeError("Evaluation budget exceeded")

        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        x = np.clip(x, self.lb, self.ub)
        y = float(func(x))

        self.n_calls += 1
        self.train_X.append(x)
        self.train_Y.append(y)

        if y < self.best_f:
            self.best_f = y
            self.best_x = x.copy()

    def _propose_next(self):
        if len(self.train_Y) < 2:
            return np.random.uniform(self.lb, self.ub, size=self.dim)

        train_X = np.asarray(self.train_X, dtype=np.float64)
        train_Y = np.asarray(self.train_Y, dtype=np.float64)

        X_norm = (train_X - self.lb) / (self.ub - self.lb)
        X_t = torch.tensor(X_norm, dtype=torch.float64)
        Y_t = torch.tensor(train_Y, dtype=torch.float64).unsqueeze(-1)

        model = SingleTaskGP(X_t, Y_t)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, maxiter=200)

        acqf = qLogExpectedImprovement(model=model, best_f=self.best_f)

        bounds_norm = torch.tensor(
            [[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64
        )

        candidate, _ = optimize_acqf(
            acqf,
            bounds=bounds_norm,
            q=1,
            num_restarts=20,
            raw_samples=128,
            options={"maxiter": 200},
        )

        cand = candidate.detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(cand)):
            raise RuntimeError("Non-finite acquisition candidate")

        x = self.lb + cand * (self.ub - self.lb)
        return np.clip(x, self.lb, self.ub)

    def __call__(self, func):
        self.lb = np.asarray(func.bounds.lb, dtype=np.float64)
        self.ub = np.asarray(func.bounds.ub, dtype=np.float64)

        if self.lb.shape != (self.dim,) or self.ub.shape != (self.dim,):
            raise ValueError("func.bounds.lb/ub must have shape (dim,)")

        for x in self.initial_points:
            self._evaluate(func, x)

        while self.n_calls < self.budget:
            try:
                x_next = self._propose_next()
            except Exception:
                x_next = np.random.uniform(self.lb, self.ub, size=self.dim)

            self._evaluate(func, x_next)

        return self.best_f, self.best_x