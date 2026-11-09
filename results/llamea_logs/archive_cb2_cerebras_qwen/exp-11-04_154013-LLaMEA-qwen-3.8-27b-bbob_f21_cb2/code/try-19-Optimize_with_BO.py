import numpy as np
import torch
import warnings
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.kernels import MaternKernel
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
        f = float(func(np.array(x, dtype=np.float64)))
        self.X.append(np.array(x, dtype=np.float64))
        self.y.append(f)
        improved = f < self.best_f
        if improved:
            self.best_f = f
            self.best_x = np.array(x, dtype=np.float64).copy()
        return f, improved

    def _fit_gp(self, dim):
        """Fit SingleTaskGP with robust multi-attempt, short Matern-5/2."""
        X_tensor = torch.tensor(np.array(self.X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1)

        best_model = None
        best_mll = -np.inf

        for attempt in range(6):
            try:
                torch.manual_seed(self.random_seed + self.n_calls * 1000 + attempt * 7919)
                model = SingleTaskGP(
                    X=X_tensor,
                    y=y_tensor,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                )
                # Short lengthscales suited for multimodal / peaky landscapes
                model.covar_module.base_kernel.lengthscale = torch.full((dim,), 0.32)
                # Small fixed noise for deterministic objective
                model.likelihood.noise = 1e-4

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_gpytorch_mll(model, max_iter=80)

                mll_val = model.mll().item()
                if mll_val > best_mll:
                    best_mll = mll_val
                    best_model = model.clone()
            except Exception:
                continue

        return best_model

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        rng = np.random.RandomState(self.random_seed + 42)

        dim = self.dim
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        bounds = torch.stack([
            torch.tensor(lb, dtype=torch.float32),
            torch.tensor(ub, dtype=torch.float32),
        ])

        # ---- Phase 1: evaluate supplied initial points ----
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # ---- Phase 2: Bayesian optimisation loop ----
        patience = 0
        PATIENCE_MAX = 5
        MIN_PTS_FOR_GP = dim + 2  # need at least dim+2 points

        while self.n_calls < self.budget:
            n_pts = len(self.X)
            progress = self.n_calls / self.budget  # fraction of budget used

            # ---- Patience: force a far-away random point ----
            if patience >= PATIENCE_MAX:
                x_new = rng.uniform(lb, ub)
                best_dist = np.linalg.norm(x_new - self.best_x)
                for _ in range(80):
                    cand = rng.uniform(lb, ub)
                    d = np.linalg.norm(cand - self.best_x)
                    if d > best_dist:
                        best_dist = d
                        x_new = cand
                f, improved = self._evaluate(func, x_new)
                patience = 0 if improved else patience + 1
                continue

            # ---- Not enough points yet: pure random ----
            if n_pts < MIN_PTS_FOR_GP:
                x_new = rng.uniform(lb, ub)
                f, improved = self._evaluate(func, x_new)
                patience = 0 if improved else patience + 1
                continue

            # ---- Fit GP surrogate ----
            model = self._fit_gp(dim)
            if model is None:
                x_new = rng.uniform(lb, ub)
                f, improved = self._evaluate(func, x_new)
                patience = 0 if improved else patience + 1
                continue

            best_f_t = torch.tensor([self.best_f], dtype=torch.float32)

            # ---- Acquisition: progressive UCB → LogEI ----
            if progress < 0.55:
                # Exploration: beta decays from 12 → 2
                t = min(progress / 0.55, 1.0)
                beta = 12.0 * (1.0 - t) ** 1.3 + 2.0
                acq_f = qUpperConfidenceBound(
                    model=model,
                    beta=max(beta, 1.5),
                    q=1,
                )
            else:
                # Exploitation: LogEI
                acq_f = qLogExpectedImprovement(
                    model=model,
                    best_f=best_f_t,
                    q=1,
                )

            # ---- Maximise acquisition with multi-start L-CMA-ES ----
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=bounds,
                        q=1,
                        num_restarts=30,
                        raw_samples=512,
                    )
                x_new = candidates.detach().numpy().squeeze()
                if x_new.ndim == 0:
                    x_new = x_new.reshape(1)
                x_new = np.clip(x_new, lb, ub)
            except Exception:
                x_new = rng.uniform(lb, ub)

            f, improved = self._evaluate(func, x_new)
            patience = 0 if improved else patience + 1

        return self.best_f, self.best_x