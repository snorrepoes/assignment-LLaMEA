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
        x = np.array(x, dtype=np.float64).flatten()
        f = func(x)
        self.X.append(x.copy())
        self.y.append(f)
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _random_points(self, lb, ub, n=1):
        return np.random.uniform(lb, ub, size=(n, self.dim))

    def _fit_gp_single(self, dim, lengthscale_init):
        X_tensor = torch.tensor(np.array(self.X), dtype=torch.float32)
        # Negate y so GP maximizes (BoTorch convention)
        y_tensor = torch.tensor(-np.array(self.y), dtype=torch.float32).unsqueeze(-1)

        model = SingleTaskGP(
            X=X_tensor,
            y=y_tensor,
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
        model.covar_module.base_kernel.lengthscale = torch.full((dim,), lengthscale_init, dtype=torch.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(model, max_iter=200)

        return model

    def _fit_gp_robust(self, dim, n_models=5):
        best_model = None
        best_mll = -np.inf
        # Diverse lengthscale initializations to escape local optima in hyperparameter space
        ls_options = [0.3, 0.5, 0.7, 0.4, 0.6]

        for i in range(min(n_models, len(ls_options))):
            seed = self.random_seed + i * 7919
            try:
                torch.manual_seed(seed)
                model = self._fit_gp_single(dim, ls_options[i])
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

        if lb.ndim == 0:
            lb = np.full(dim, float(lb))
        if ub.ndim == 0:
            ub = np.full(dim, float(ub))

        bounds = torch.stack([
            torch.tensor(lb, dtype=torch.float32),
            torch.tensor(ub, dtype=torch.float32)
        ])

        # ---- Phase 1: Evaluate initial points ----
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.last_best_f(), self.last_best_x()

        # ---- Phase 2: Bayesian Optimization loop ----
        no_improve_count = 0
        best_f_reference = self.best_f

        while self.n_calls < self.budget:
            n_points = len(self.X)
            remaining = self.budget - self.n_calls

            # Adaptive batch size
            if remaining == 1:
                q = 1
            elif n_points < dim + 2:
                q = min(2, remaining)
            else:
                q = min(3, remaining)

            # Too few points for meaningful GP — random exploration
            if n_points < dim + 1:
                pts = self._random_points(lb, ub, q)
                for i in range(q):
                    if self.n_calls < self.budget:
                        self._evaluate(func, pts[i])
                continue

            # Patience-based random restart (explore new region)
            if no_improve_count >= 12:
                pts = self._random_points(lb, ub, q)
                for i in range(q):
                    if self.n_calls >= self.budget:
                        break
                    self._evaluate(func, pts[i])
                no_improve_count = 0
                best_f_reference = self.best_f
                continue

            # Fit GP with diverse hyperparameter initialization
            model = self._fit_gp_robust(dim)

            if model is None:
                pts = self._random_points(lb, ub, q)
                for i in range(q):
                    if self.n_calls < self.budget:
                        self._evaluate(func, pts[i])
                continue

            # best_f in negated space: max(-y) = -min(y)
            best_f_tensor = torch.tensor([-self.best_f], dtype=torch.float32)

            # Progressive acquisition: heavy exploration → exploitation
            progress = n_points / self.budget
            if progress < 0.5:
                # Strong exploration: beta decreases from 8 to 3
                beta = 8.0 - 10.0 * progress
                beta = max(beta, 3.0)
                acq_f = qUpperConfidenceBound(model=model, beta=beta, q=q)
            elif progress < 0.75:
                # Moderate exploration
                beta = 2.5
                acq_f = qUpperConfidenceBound(model=model, beta=beta, q=q)
            else:
                # Exploitation
                acq_f = qLogExpectedImprovement(model=model, best_f=best_f_tensor, q=q)

            # Multi-start acquisition optimization
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=bounds,
                        q=q,
                        num_restarts=12,
                        raw_samples=512,
                    )
            except Exception:
                pts = self._random_points(lb, ub, q)
                for i in range(q):
                    if self.n_calls < self.budget:
                        self._evaluate(func, pts[i])
                continue

            candidates = candidates.detach().numpy()

            # Evaluate the suggested candidates
            for i in range(min(q, candidates.shape[0])):
                if self.n_calls >= self.budget:
                    break
                x_new = np.clip(candidates[i], lb, ub)
                self._evaluate(func, x_new)

            # Update patience
            if self.best_f < best_f_reference - 1e-10:
                no_improve_count = 0
                best_f_reference = self.best_f
            else:
                no_improve_count += 1

        return self.best_f, self.best_x

    def last_best_f(self):
        return self.best_f

    def last_best_x(self):
        return self.best_x