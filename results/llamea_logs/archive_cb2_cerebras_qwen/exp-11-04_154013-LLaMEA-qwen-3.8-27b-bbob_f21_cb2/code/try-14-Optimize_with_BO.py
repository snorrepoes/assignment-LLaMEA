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
from scipy.stats import qmc
from scipy.spatial.distance import cdist


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
            raise RuntimeError(f"Budget exceeded: {self.n_calls} > {self.budget}")
        x = np.array(x, dtype=np.float64).flatten()
        f = float(func(x))
        self.X.append(x.copy())
        self.y.append(f)
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _fit_gp(self):
        X_tensor = torch.tensor(np.array(self.X), dtype=torch.float)
        y_tensor = torch.tensor(np.array(self.y), dtype=torch.float).unsqueeze(-1)

        model = SingleTaskGP(
            X=X_tensor,
            y=y_tensor,
            input_transform=Normalize(d=self.dim),
            outcome_transform=Standardize(m=1),
        )
        model.covar_module.base_kernel.lengthscale = torch.full((self.dim,), 0.1)
        model.likelihood.noise = 0.001

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(model, progress_bar=False)

        return model

    def _is_near_existing(self, x, threshold=0.05):
        if not self.X:
            return False
        X_arr = np.array(self.X)
        d = np.min(np.linalg.norm(X_arr - x, axis=1))
        return d < threshold

    def _diversify_point(self, lb, ub):
        if len(self.X) == 0:
            return np.random.uniform(lb, ub, size=self.dim)

        X_arr = np.array(self.X)
        rng = np.random.RandomState(self.random_seed + self.n_calls * 13 + 7)

        n = 128
        candidates = np.zeros((n, self.dim))
        sampler = qmc.Sobol(d=self.dim, seed=rng.randint(2**31))
        sobol = sampler.random(n // 2)
        candidates[:n // 2] = sobol
        candidates[n // 2:] = rng.random((n - n // 2, self.dim))

        candidates = lb + candidates * (ub - lb)

        # Add some boundary-biased points
        for i in range(8):
            idx = n // 2 + i
            pt = rng.uniform(lb, ub, size=self.dim)
            for j in range(self.dim):
                if rng.random() < 0.3:
                    if rng.random() < 0.5:
                        pt[j] = lb[j] + 0.1 * (ub[j] - lb[j]) * rng.random()
                    else:
                        pt[j] = ub[j] - 0.1 * (ub[j] - lb[j]) * rng.random()
            candidates[idx] = pt

        denom = (ub - lb).max()
        X_norm = (X_arr - lb) / denom
        C_norm = (candidates - lb) / denom
        d = cdist(C_norm, X_norm, metric='euclidean')
        min_d = d.min(axis=1)
        return candidates[np.argmax(min_d)]

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        dim = self.dim
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        bounds = [torch.tensor(lb, dtype=torch.float), torch.tensor(ub, dtype=torch.float)]

        # Phase 1: Evaluate initial points
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # Phase 2: Bayesian Optimization
        n_init = len(self.initial_points)
        remaining = self.budget - n_init
        bo_step = 0
        patience = 0
        max_patience = 6

        while self.n_calls < self.budget:
            bo_step += 1
            progress = min(bo_step / max(remaining, 1), 1.0)

            # Patience-based far-point diversification
            if patience >= max_patience:
                x_new = self._diversify_point(lb, ub)
                old_best = self.best_f
                self._evaluate(func, x_new)
                if self.best_f < old_best - 1e-8:
                    patience = 0
                continue

            # Fit GP surrogate
            try:
                model = self._fit_gp()
            except Exception:
                x_new = self._diversify_point(lb, ub)
                self._evaluate(func, x_new)
                patience += 1
                continue

            best_f_tensor = torch.tensor([self.best_f], dtype=torch.float)

            # Adaptive acquisition: UCB with linearly decreasing beta, then EI
            if progress < 0.75:
                beta = 10.0 - (10.0 - 1.5) * (progress / 0.75)
                acq_f = qUpperConfidenceBound(model=model, beta=beta, q=1)
            else:
                acq_f = qLogExpectedImprovement(model=model, best_f=best_f_tensor, q=1)

            # Optimize acquisition with multi-restart
            x_new = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=bounds,
                        q=1,
                        num_restarts=25,
                        raw_samples=256,
                    )
                x_new = candidates.numpy().flatten()
            except Exception:
                pass

            # Near-duplicate rejection: fall back to diversification
            if x_new is None or self._is_near_existing(x_new):
                x_new = self._diversify_point(lb, ub)

            old_best = self.best_f
            self._evaluate(func, x_new)

            if self.best_f < old_best - 1e-8:
                patience = 0
            else:
                patience += 1

        return self.best_f, self.best_x