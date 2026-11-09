import numpy as np
import torch
import warnings
import math
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.optim import optimize_acqf


def _cosine_warp_torch(X, lb, ub):
    X_mapped = math.pi * (X - lb) / (ub - lb)
    return torch.cat([torch.sin(X_mapped), torch.cos(X_mapped)], dim=-1)


def _cosine_warp_numpy(X, lb, ub):
    X_mapped = math.pi * (X - lb) / (ub - lb)
    return np.concatenate([np.sin(X_mapped), np.cos(X_mapped)], axis=1)


class _WarpedGP(torch.nn.Module):
    def __init__(self, model, lb, ub):
        super().__init__()
        self.model = model
        self.lb = lb
        self.ub = ub

    def posterior(self, X, **kwargs):
        X_w = _cosine_warp_torch(X, self.lb, self.ub)
        return self.model.posterior(X_w, **kwargs)

    @property
    def num_outputs(self):
        return self.model.num_outputs

    @property
    def train_inputs(self):
        return self.model.train_inputs


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

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        dim = self.dim
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        bounds = torch.stack([torch.tensor(lb, dtype=torch.float),
                              torch.tensor(ub, dtype=torch.float)])
        lb_t = torch.tensor(lb, dtype=torch.float)
        ub_t = torch.tensor(ub, dtype=torch.float)

        # Phase 1: Evaluate initial points
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # Phase 2: Bayesian Optimization loop
        batch_size = 4
        iter_count = 0
        prev_best = self.best_f
        no_improve = 0

        while self.n_calls < self.budget:
            q = min(batch_size, self.budget - self.n_calls)
            n = len(self.X)
            iter_count += 1

            # Not enough data for meaningful GP -> random
            if n < dim + 2:
                for _ in range(q):
                    if self.n_calls >= self.budget:
                        break
                    self._evaluate(func, np.random.uniform(lb, ub))
                continue

            # Periodic random exploration every 4th iteration for diversity
            if iter_count % 4 == 0 and q > 2:
                self._evaluate(func, np.random.uniform(lb, ub))
                q -= 1
                if q <= 0:
                    continue

            # Escape from local optima: extra random point when stuck
            if no_improve >= 4 and q > 1:
                self._evaluate(func, np.random.uniform(lb, ub))
                q -= 1
                no_improve = 0
                if q <= 0:
                    continue

            # Fit GP on cosine-warped inputs
            X_np = np.array(self.X)
            y_np = np.array(self.y)
            X_w_np = _cosine_warp_numpy(X_np, lb, ub)
            wdim = 2 * dim

            X_t = torch.tensor(X_w_np, dtype=torch.float)
            y_t = torch.tensor(y_np, dtype=torch.float).unsqueeze(-1)

            model = None
            for _ in range(3):
                try:
                    model = SingleTaskGP(
                        X=X_t, y=y_t,
                        input_transform=Normalize(d=wdim),
                        outcome_transform=Standardize(m=1),
                    )
                    model.covar_module.base_kernel.lengthscale = torch.full((wdim,), 1.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit_gpytorch_mll(model)
                    break
                except Exception:
                    model = None

            if model is None:
                for _ in range(q):
                    if self.n_calls >= self.budget:
                        break
                    self._evaluate(func, np.random.uniform(lb, ub))
                continue

            # Acquisition in original 6D space via warped model wrapper
            warped_model = _WarpedGP(model, lb_t, ub_t)
            best_f_t = torch.tensor([self.best_f], dtype=torch.float)

            if n <= 2 * dim:
                acq_f = qUpperConfidenceBound(model=warped_model, beta=2.5, q=q)
            else:
                acq_f = qLogExpectedImprovement(
                    model=warped_model, best_f=best_f_t, q=q, monte_carlo_samples=256
                )

            # Multi-restart acquisition optimization
            cands = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cands = optimize_acqf(
                        f=acq_f,
                        bounds=bounds,
                        q=q,
                        num_restarts=15,
                        raw_samples=256,
                        batch_initial_points=15,
                    )
            except Exception:
                pass

            # Evaluate selected candidates
            if cands is not None:
                cands_np = cands.detach().numpy()
                for i in range(q):
                    if self.n_calls >= self.budget:
                        break
                    self._evaluate(func, cands_np[i])
            else:
                for _ in range(q):
                    if self.n_calls >= self.budget:
                        break
                    self._evaluate(func, np.random.uniform(lb, ub))

            # Update tracking
            if self.best_f < prev_best - 1e-10:
                no_improve = 0
            else:
                no_improve += 1
            prev_best = self.best_f

        return self.best_f, self.best_x