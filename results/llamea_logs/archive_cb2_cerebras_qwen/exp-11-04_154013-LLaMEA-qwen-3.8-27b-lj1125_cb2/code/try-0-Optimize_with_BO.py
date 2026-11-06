import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.qucb import qUpperConfidenceBound
from botorch.optim import optimize_acqf


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_calls = 0
        self.X_all = []
        self.f_all = []
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        lb = np.array(func.bounds.lb, dtype=np.float64)
        ub = np.array(func.bounds.ub, dtype=np.float64)

        # Evaluate initial points (count against budget)
        for x in self.initial_points:
            if self.n_calls >= self.budget:
                raise Exception("Budget exceeded during initial evaluations")
            x = np.array(x, dtype=np.float64).flatten()
            f = float(func(x))
            self.n_calls += 1
            self.X_all.append(x)
            self.f_all.append(f)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x.copy()

        remaining = self.budget - self.n_calls

        # Input normalization: scale to [0, 1]^d based on true bounds
        limits = torch.tensor(np.stack([lb, ub]), dtype=torch.float32)
        input_transform = Normalize(self.dim, limits=limits)

        for it in range(remaining):
            X = np.array(self.X_all, dtype=np.float64)
            f = np.array(self.f_all, dtype=np.float64)

            # Log transform to compress orders of magnitude
            f_log = np.log(np.maximum(f, 1e-300))

            X_torch = torch.tensor(X, dtype=torch.float32)
            Y_torch = torch.tensor(f_log.reshape(-1, 1), dtype=torch.float32)

            # Build and fit GP surrogate
            model = SingleTaskGP(
                X=X_torch,
                Y=Y_torch,
                input_transform=input_transform,
                outcome_transform=Standardize(1, min_stdev=0.01),
            )
            model.mle()

            # Adaptive beta: strong exploration early, transition to exploitation
            progress = it / max(remaining - 1, 1)
            beta = 1.0 + 4.0 * (1.0 - progress)  # 5.0 → 1.0

            acqf = qUpperConfidenceBound(model=model, beta=beta)

            # Optimize acquisition in normalized [0, 1]^d space
            bounds = torch.stack([
                torch.zeros(self.dim, dtype=torch.float32),
                torch.ones(self.dim, dtype=torch.float32),
            ])

            candidate, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=16,
                raw_samples=512,
                options=dict(maxiter=200),
            )

            # Map back to original parameter space
            x_new = (candidate.numpy().flatten() * (ub - lb) + lb).astype(np.float64)
            x_new = np.clip(x_new, lb, ub)

            # Evaluate the new point
            if self.n_calls >= self.budget:
                raise Exception("Budget exceeded")
            f_new = float(func(x_new))
            self.n_calls += 1
            self.X_all.append(x_new)
            self.f_all.append(f_new)

            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new.copy()

        return self.f_opt, self.x_opt