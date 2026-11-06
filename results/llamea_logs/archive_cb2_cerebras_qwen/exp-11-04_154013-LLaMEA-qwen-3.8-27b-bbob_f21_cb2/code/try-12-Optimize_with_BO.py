import numpy as np
import torch
import warnings
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
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
        self.stagnation_count = 0

    def _evaluate(self, func, x):
        self.n_calls += 1
        if self.n_calls > self.budget:
            raise Exception(f"Budget exceeded: call {self.n_calls} > budget {self.budget}")
        f = float(func(np.array(x, dtype=np.float64)))
        self.X.append(np.array(x, dtype=np.float64))
        self.y.append(f)
        if f < self.best_f:
            self.best_f = f
            self.best_x = np.array(x, dtype=np.float64).copy()
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        return f

    def _random_point(self, lb, ub):
        return np.random.uniform(lb, ub)

    def _fit_gp_robust(self):
        """Fit SingleTaskGP with diverse lengthscale initializations, return best MLL model."""
        X_tensor = torch.tensor(np.array(self.X), dtype=torch.float)
        y_tensor = torch.tensor(np.array(self.y), dtype=torch.float).unsqueeze(-1)
        dim = self.dim

        best_model = None
        best_mll = -np.inf

        # Diverse lengthscale scales to capture local multimodal structure
        ls_scales = [0.15, 0.25, 0.4, 0.7]

        for i, ls in enumerate(ls_scales):
            try:
                torch.manual_seed(self.random_seed + i * 104729 + 3)
                model = SingleTaskGP(
                    X=X_tensor,
                    y=y_tensor,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                )
                # Short initial lengthscales encourage local fitting for multimodal functions
                model.covar_module.base_kernel.lengthscale = torch.full((dim,), ls)
                model.covar_module.outputscale = torch.tensor(5.0)
                model.noise = 1e-4

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_gpytorch_mll(model, max_iter=150)

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

        # ---- Phase 2: Bayesian Optimization with q=2 ----
        q = 2
        patience = 4  # BO iterations without improvement before aggressive exploration

        while self.n_calls < self.budget:
            n_points = len(self.X)
            progress = self.n_calls / self.budget

            # Need enough points for meaningful GP fitting
            min_points = dim + 2
            if n_points < min_points:
                n_evaluate = min(q, self.budget - self.n_calls)
                for _ in range(n_evaluate):
                    x_new = self._random_point(lb, ub)
                    self._evaluate(func, x_new)
                continue

            # Fit GP surrogate with diverse restarts
            model = self._fit_gp_robust()

            if model is None:
                # Fallback: random exploration
                n_evaluate = min(q, self.budget - self.n_calls)
                for _ in range(n_evaluate):
                    x_new = self._random_point(lb, ub)
                    self._evaluate(func, x_new)
                continue

            best_f_tensor = torch.tensor([self.best_f], dtype=torch.float)

            # Adaptive acquisition selection based on progress and stagnation
            if self.stagnation_count >= patience:
                # Stagnation detected: aggressive global exploration
                beta = 12.0
                acq_f = qUpperConfidenceBound(model=model, beta=beta, q=q)
            elif progress < 0.2:
                # Early phase: strong exploration
                beta = 6.0
                acq_f = qUpperConfidenceBound(model=model, beta=beta, q=q)
            elif progress < 0.45:
                # Mid phase: decreasing exploration
                beta = 4.0 - 8.0 * (progress - 0.2) / 0.25
                acq_f = qUpperConfidenceBound(model=model, beta=max(beta, 1.5), q=q)
            elif progress < 0.7:
                # Late-mid: transition to exploitation
                beta = 1.5 - 2.5 * (progress - 0.45) / 0.25
                acq_f = qUpperConfidenceBound(model=model, beta=max(beta, 0.5), q=q)
            else:
                # Late phase: pure exploitation
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
                        num_restarts=30,
                        raw_samples=1024,
                    )
            except Exception:
                n_evaluate = min(q, self.budget - self.n_calls)
                for _ in range(n_evaluate):
                    x_new = self._random_point(lb, ub)
                    self._evaluate(func, x_new)
                continue

            candidates = candidates.detach().numpy()

            # Evaluate the batch
            for j in range(q):
                if self.n_calls >= self.budget:
                    break
                x_new = np.clip(candidates[j], lb, ub)
                self._evaluate(func, x_new)

        return self.best_f, self.best_x