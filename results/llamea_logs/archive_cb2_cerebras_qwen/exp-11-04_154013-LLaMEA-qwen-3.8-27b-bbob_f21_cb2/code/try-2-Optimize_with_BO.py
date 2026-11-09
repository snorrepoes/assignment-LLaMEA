import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from scipy.stats import qmc


class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_evals = 0
        self._f_opt = np.inf
        self._x_opt = None

    def _evaluate(self, x, func):
        x = np.asarray(x, dtype=np.float64)
        self.n_evals += 1
        if self.n_evals > self.budget:
            raise RuntimeError(f"Evaluation budget of {self.budget} exceeded.")
        f = float(func(x))
        if f < self._f_opt:
            self._f_opt = f
            self._x_opt = x.copy()
        return f

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)

        # Seed all stochastic components
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        X_list = []
        f_list = []

        # Phase 1: Initial design
        # 1a. Evaluate provided initial points (count against budget)
        if self.initial_points is not None and len(self.initial_points) > 0:
            init = np.asarray(self.initial_points, dtype=np.float64)
            n_avail = min(len(init), self.budget)
            for x in init[:n_avail]:
                f = self._evaluate(x, func)
                X_list.append(x.copy())
                f_list.append(f)

        # 1b. Complement with Sobol points for space-filling coverage
        target_init = min(12, self.budget)
        if len(X_list) < target_init:
            n_needed = target_init - len(X_list)
            sobol_gen = qmc.Sobol(d=self.dim, scramble=True, seed=self.random_seed)
            points = sobol_gen.random(n_needed)
            points = points * (ub - lb) + lb
            for x in points:
                f = self._evaluate(x, func)
                X_list.append(x.copy())
                f_list.append(f)

        X = np.array(X_list)
        f = np.array(f_list)

        # Phase 2: Bayesian Optimization loop
        no_improve = 0
        patience = 10
        total_bo_steps = self.budget - self.n_evals
        bo_step = 0

        while self.n_evals < self.budget:
            bo_step += 1
            progress = bo_step / max(total_bo_steps, 1)

            if no_improve >= patience:
                # Exploration restart: pick a point far from all existing
                best_x = np.random.uniform(lb, ub)
                best_dist = -np.inf
                for _ in range(200):
                    x_cand = np.random.uniform(lb, ub)
                    min_d = np.min(np.linalg.norm(X - x_cand, axis=1))
                    if min_d > best_dist:
                        best_dist = min_d
                        best_x = x_cand

                x_next = best_x
                f_next = self._evaluate(x_next, func)
                no_improve = 0
            else:
                # Normal BO step
                # Normalize inputs to [0,1]
                X_norm = (X - lb) / (ub - lb)

                # Standardize outputs
                f_mean = f.mean()
                f_std = f.std()
                if f_std < 1e-10:
                    f_z = f - f_mean
                else:
                    f_z = (f - f_mean) / f_std

                X_t = torch.tensor(X_norm, dtype=torch.double)
                f_t = torch.tensor(f_z, dtype=torch.double).unsqueeze(-1)

                # Build and fit GP surrogate
                model = SingleTaskGP(X_t, f_t, noisy_observation=False)
                model.covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=self.dim)
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                try:
                    fit_gpytorch_mll(mll)
                except Exception:
                    pass

                f_best = f_t.min().item()

                # Acquisition: UCB (exploration) early, LogEI (exploitation) later
                if progress < 0.5:
                    beta = 2.0 + 4.0 * (1.0 - 2.0 * progress)  # range ~6→2
                    acq = qUpperConfidenceBound(model=model, beta=beta)
                else:
                    acq = qLogExpectedImprovement(model=model, best_f=f_best)

                bounds = torch.tensor(
                    [[0.0] * self.dim, [1.0] * self.dim], dtype=torch.double
                )

                n_restarts = max(10, self.dim * 2)
                best_candidate, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=n_restarts,
                    raw_samples=512,
                )

                x_next_norm = best_candidate.detach().cpu().numpy().flatten()
                x_next = x_next_norm * (ub - lb) + lb
                x_next = np.clip(x_next, lb, ub)

                # Avoid duplicate evaluations
                if len(X) > 0 and np.any(np.all(np.abs(X - x_next) < 1e-8, axis=1)):
                    x_next = x_next + np.random.normal(0, 0.005 * (ub - lb), size=self.dim)
                    x_next = np.clip(x_next, lb, ub)

                f_next = self._evaluate(x_next, func)

                if f_next < self._f_opt:
                    no_improve = 0
                else:
                    no_improve += 1

            # Update data
            X = np.vstack([X, x_next.reshape(1, -1)])
            f = np.append(f, f_next)

        return self._f_opt, self._x_opt