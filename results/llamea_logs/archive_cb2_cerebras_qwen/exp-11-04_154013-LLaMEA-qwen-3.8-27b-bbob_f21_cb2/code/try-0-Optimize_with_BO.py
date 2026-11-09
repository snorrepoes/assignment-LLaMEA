import numpy as np
import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import InputStandardize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf
from scipy.stats import qmc


class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_calls = 0
        self.x_best = None
        self.f_best = np.inf
        self.X_obs = []
        self.f_obs = []

    def _generate_sobol(self, lb, ub, n):
        """Generate n Sobol quasi-random points in [lb, ub]."""
        sampler = qmc.Sobol(d=self.dim, seed=self.random_seed)
        pts = sampler.random(n).astype(np.float64)
        pts = pts * (ub.numpy() - lb.numpy()) + lb.numpy()
        return np.clip(pts, lb.numpy(), ub.numpy())

    def _fit_gp(self, X, F, seed_offset=0):
        """Fit a SingleTaskGP with retry and best-model selection."""
        best_model = None
        best_val = -np.inf

        for attempt in range(3):
            try:
                torch.manual_seed(self.random_seed + seed_offset + attempt)
                model = SingleTaskGP(
                    X, F,
                    input_transform=InputStandardize(self.dim),
                    outcome_transform=Standardize(m=1),
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll, iterations=200)

                with torch.no_grad():
                    val = mll(model(X, F)).item()
                if val > best_val:
                    best_val = val
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            # Fallback: use a GP with default hyperparameters
            model = SingleTaskGP(X, F)
            model.covar_module.base_kernel.lengthscale = torch.ones(
                self.dim, 1, dtype=torch.float64
            ).exp()
            model.likelihood.noise = torch.tensor(1e-4, dtype=torch.float64)
            best_model = model
        return best_model

    def _optimize_acquisition(self, acqf, bounds, lb, ub, seed_offset):
        """Optimize the acquisition function with multiple restarts and fallback."""
        rng = torch.Generator().manual_seed(self.random_seed + seed_offset)
        try:
            candidates, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=15,
                raw_samples=2048,
                options={"batch_initial_candidates": 15, "maxiter": 200},
                rng=rng,
            )
            return candidates.detach().numpy()[0].astype(np.float64)
        except Exception:
            # Fallback: sample in a local region around best point
            span = (ub.numpy() - lb.numpy())
            local_lb = np.maximum(lb.numpy(), self.x_best - 0.15 * span)
            local_ub = np.minimum(ub.numpy(), self.x_best + 0.15 * span)
            x = np.random.uniform(local_lb, local_ub)
            return x.astype(np.float64)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        lb = torch.tensor(np.array(func.bounds.lb, dtype=np.float64))
        ub = torch.tensor(np.array(func.bounds.ub, dtype=np.float64))
        bounds = torch.stack([lb, ub])

        # ---- Phase 1: Initial Design ----
        if self.initial_points is not None and len(self.initial_points) > 0:
            initial_pts = np.array(self.initial_points, dtype=np.float64)
            initial_pts = np.clip(initial_pts, lb.numpy(), ub.numpy())
        else:
            n_init = min(max(10, self.budget // 5), self.budget)
            initial_pts = self._generate_sobol(lb, ub, n_init)

        for x in initial_pts:
            if self.n_calls >= self.budget:
                raise RuntimeError(
                    f"Budget exceeded: {self.n_calls} >= {self.budget}"
                )
            x_eval = x.astype(np.float64)
            f_val = float(func(x_eval))
            self.n_calls += 1
            self.X_obs.append(x_eval)
            self.f_obs.append(f_val)
            if f_val < self.f_best:
                self.f_best = f_val
                self.x_best = x_eval.copy()

        # ---- Phase 2: Bayesian Optimization Loop ----
        n_bo = self.budget - self.n_calls
        # First ~20% of BO iterations: explore (UCB), rest: exploit (EI)
        n_explore = max(3, int(0.2 * n_bo))

        for i in range(n_bo):
            X = torch.tensor(np.array(self.X_obs), dtype=torch.float64)
            F = torch.tensor(np.array(self.f_obs), dtype=torch.float64).unsqueeze(-1)

            # Fit surrogate
            model = self._fit_gp(X, F, seed_offset=i)

            f_min = F.min().item()

            # Select acquisition function
            if i < n_explore:
                # Exploration: UCB with increasing beta for better coverage
                beta = 1.0 + 0.5 * (i + 1)
                acqf = qUpperConfidenceBound(
                    model=model, beta=beta, q=1
                )
            else:
                # Exploitation: Expected Improvement
                acqf = qLogExpectedImprovement(
                    model=model,
                    best_f=torch.tensor(f_min, dtype=torch.float64),
                    q=1,
                )

            # Optimize acquisition → next candidate
            x_new = self._optimize_acquisition(
                acqf, bounds, lb, ub, seed_offset=self.n_calls + i
            )
            x_new = np.clip(x_new, lb.numpy(), ub.numpy())

            # Evaluate
            f_new = float(func(x_new))
            self.n_calls += 1
            self.X_obs.append(x_new)
            self.f_obs.append(f_new)

            if f_new < self.f_best:
                self.f_best = f_new
                self.x_best = x_new.copy()

        return self.f_best, self.x_best