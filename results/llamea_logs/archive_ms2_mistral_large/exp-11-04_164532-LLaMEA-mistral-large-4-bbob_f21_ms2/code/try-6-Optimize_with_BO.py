import numpy as np
import torch
import random
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        if self.initial_points.ndim == 1:
            self.initial_points = self.initial_points.reshape(1, -1)
        self.random_seed = int(random_seed)

        torch.set_default_dtype(torch.float64)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        dim = self.dim
        budget = self.budget

        def to_unit(x):
            return (np.asarray(x, dtype=np.float64) - lb) / (ub - lb)

        def from_unit(x_unit):
            return lb + np.asarray(x_unit, dtype=np.float64) * (ub - lb)

        calls = 0

        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                raise RuntimeError("Evaluation budget exceeded")
            calls += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        X_list = []
        Y_list = []

        # Evaluate provided initial DoE points first.
        for x in self.initial_points:
            y = safe_func(x)
            X_list.append(np.asarray(x, dtype=np.float64))
            Y_list.append(y)

        if len(X_list) == 0:
            x = np.random.uniform(lb, ub, size=dim)
            y = safe_func(x)
            X_list.append(x)
            Y_list.append(y)

        X = np.array(X_list, dtype=np.float64).reshape(-1, dim)
        Y = np.array(Y_list, dtype=np.float64).reshape(-1, 1)

        best_idx = int(np.argmin(Y))
        best_f = float(Y[best_idx, 0])
        best_x = X[best_idx].copy()

        bounds_unit = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float64)

        while calls < budget:
            train_X = torch.tensor(to_unit(X), dtype=torch.float64)
            train_Y = torch.tensor(Y, dtype=torch.float64)

            x_unit = None
            try:
                covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim))
                model = SingleTaskGP(train_X, train_Y, covar_module=covar_module)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                acq = qLogExpectedImprovement(
                    model=model,
                    best_f=best_f,
                    X_baseline=train_X,
                )

                candidate, _ = optimize_acqf(
                    acq,
                    bounds=bounds_unit,
                    q=1,
                    num_restarts=20,
                    raw_samples=1024,
                    options={"maxiter": 200},
                )

                x_unit = candidate.detach().cpu().numpy().reshape(-1)
                if not np.all(np.isfinite(x_unit)):
                    x_unit = None
            except Exception:
                x_unit = None

            if x_unit is None:
                x_unit = np.random.uniform(0.0, 1.0, size=dim)

            x_unit = np.clip(x_unit, 0.0, 1.0)
            x = from_unit(x_unit)
            y = safe_func(x)

            X_list.append(x)
            Y_list.append(y)

            X = np.array(X_list, dtype=np.float64).reshape(-1, dim)
            Y = np.array(Y_list, dtype=np.float64).reshape(-1, 1)

            if y < best_f:
                best_f = float(y)
                best_x = x.copy()

        return best_f, best_x