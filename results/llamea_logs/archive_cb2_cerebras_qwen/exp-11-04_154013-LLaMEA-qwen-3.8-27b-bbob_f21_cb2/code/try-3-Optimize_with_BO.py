import numpy as np
import torch
import warnings
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.optim import optimize_acqf


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = np.array(initial_points, dtype=np.float64)
        self.random_seed = random_seed
        self.n_calls = 0
        self.best_f = np.inf
        self.best_x = None
        self.X = []
        self.y = []

    def _evaluate(self, func, x):
        """Evaluate a single point and update bookkeeping."""
        self.n_calls += 1
        if self.n_calls > self.budget:
            raise Exception(f"Budget exceeded: call {self.n_calls} > budget {self.budget}")
        f = func(np.array(x, dtype=np.float64))
        self.X.append(np.array(x, dtype=np.float64))
        self.y.append(f)
        if f < self.best_f:
            self.best_f = f
            self.best_x = np.array(x, dtype=np.float64).copy()
        return f

    def _random_point(self, lb, ub):
        return np.random.uniform(lb, ub)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        dim = self.dim
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        bounds = torch.stack([torch.tensor(lb, dtype=torch.float),
                              torch.tensor(ub, dtype=torch.float)])

        # ---- Phase 1: Evaluate initial points ----
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # ---- Phase 2: Bayesian Optimization loop ----
        batch_size = 3

        while self.n_calls < self.budget:
            q = min(batch_size, self.budget - self.n_calls)
            n_points = len(self.X)

            # If too few points for a meaningful GP, fall back to random
            if n_points < self.dim + 2:
                for _ in range(q):
                    if self.n_calls >= self.budget:
                        break
                    x_new = self._random_point(lb, ub)
                    self._evaluate(func, x_new)
                continue

            X_tensor = torch.tensor(np.array(self.X), dtype=torch.float)
            y_tensor = torch.tensor(np.array(self.y), dtype=torch.float).unsqueeze(-1)

            # Fit GP surrogate
            try:
                model = SingleTaskGP(
                    X=X_tensor,
                    y=y_tensor,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                )
                # Initialize lengthscales to a reasonable scale for [-5,5]^dim
                model.covar_module.base_kernel.lengthscale = torch.full((dim,), 2.0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_gpytorch_mll(model)
            except Exception:
                for _ in range(q):
                    if self.n_calls >= self.budget:
                        break
                    x_new = self._random_point(lb, ub)
                    self._evaluate(func, x_new)
                continue

            best_f_tensor = torch.tensor([self.best_f], dtype=torch.float)

            # Select acquisition: explore early, exploit later
            if n_points <= 3 * dim:
                acq_f = qUpperConfidenceBound(
                    model=model,
                    beta=2.5,
                    q=q,
                )
            else:
                acq_f = qLogExpectedImprovement(
                    model=model,
                    best_f=best_f_tensor,
                    q=q,
                )

            # Multi-start acquisition optimization
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=bounds,
                        q=q,
                        num_restarts=10,
                        raw_samples=256,
                        batch_initial_points=10,
                    )
            except Exception:
                for _ in range(q):
                    if self.n_calls >= self.budget:
                        break
                    x_new = self._random_point(lb, ub)
                    self._evaluate(func, x_new)
                continue

            # Evaluate the selected batch
            candidates_np = candidates.numpy()
            for i in range(q):
                if self.n_calls >= self.budget:
                    break
                x_new = candidates_np[i]
                self._evaluate(func, x_new)

        return self.best_f, self.best_x