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

    def _fit_gp_robust(self, dim):
        """Fit SingleTaskGP with multiple restarts, return best model."""
        X_tensor = torch.tensor(np.array(self.X), dtype=torch.float)
        y_tensor = torch.tensor(np.array(self.y), dtype=torch.float).unsqueeze(-1)

        best_model = None
        best_mll = -np.inf

        for seed_attempt in range(4):
            try:
                torch.manual_seed(self.random_seed + seed_attempt * 7919)
                model = SingleTaskGP(
                    X=X_tensor,
                    y=y_tensor,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                )
                # Initialize lengthscales wider to encourage smoother initial fit
                model.covar_module.base_kernel.lengthscale = torch.full((dim,), 0.6)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_gpytorch_mll(model, max_iter=120)

                mll_val = model.mll().item()
                if mll_val > best_mll:
                    best_mll = mll_val
                    best_model = model
            except Exception:
                continue

        return best_model

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

        # ---- Phase 2: Bayesian Optimization with q=1 ----
        while self.n_calls < self.budget:
            q = 1
            n_points = len(self.X)
            progress = self.n_calls / self.budget

            # Too few points for meaningful GP — random exploration
            if n_points < self.dim + 2:
                x_new = self._random_point(lb, ub)
                self._evaluate(func, x_new)
                continue

            # Fit GP surrogate
            model = self._fit_gp_robust(dim)

            if model is None:
                x_new = self._random_point(lb, ub)
                self._evaluate(func, x_new)
                continue

            best_f_tensor = torch.tensor([self.best_f], dtype=torch.float)

            # Adaptive acquisition: explore early, exploit late
            if progress < 0.5:
                # Exploration: beta decreases linearly from 10 to 1
                beta = 10.0 - 18.0 * progress  # at progress=0: 10, at 0.5: 1
                acq_f = qUpperConfidenceBound(
                    model=model,
                    beta=max(beta, 1.0),
                    q=q,
                )
            else:
                # Exploitation: LogEI
                acq_f = qLogExpectedImprovement(
                    model=model,
                    best_f=best_f_tensor,
                    q=q,
                )

            # Multi-start L-CMA-ES for acquisition maximization
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=bounds,
                        q=q,
                        num_restarts=25,
                        raw_samples=512,
                    )
            except Exception:
                x_new = self._random_point(lb, ub)
                self._evaluate(func, x_new)
                continue

            x_new = candidates.detach().numpy().squeeze()
            if x_new.ndim == 0:
                x_new = x_new.reshape(1)
            x_new = np.clip(x_new, lb, ub)

            self._evaluate(func, x_new)

        return self.best_f, self.best_x