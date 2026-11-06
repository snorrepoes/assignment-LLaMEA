import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
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
        x = np.asarray(x, dtype=np.float64).flatten()
        self.n_evals += 1
        if self.n_evals > self.budget:
            raise RuntimeError(f"Evaluation budget of {self.budget} exceeded.")
        f = float(func(x))
        if f < self._f_opt:
            self._f_opt = f
            self._x_opt = x.copy()
        return f

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=np.float64).flatten()
        ub = np.asarray(func.bounds.ub, dtype=np.float64).flatten()

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        X_list = []
        f_list = []

        # Phase 1: Initial design
        if self.initial_points is not None and len(self.initial_points) > 0:
            init = np.asarray(self.initial_points, dtype=np.float64)
            n_avail = min(len(init), self.budget)
            for x in init[:n_avail]:
                f = self._evaluate(x, func)
                X_list.append(x.copy().flatten())
                f_list.append(f)

        target_init = min(10, self.budget)
        if len(X_list) < target_init:
            n_needed = target_init - len(X_list)
            sobol_gen = qmc.Sobol(d=self.dim, scramble=True, seed=self.random_seed)
            points = sobol_gen.random(n_needed)
            points = points * (ub - lb) + lb
            for x in points:
                f = self._evaluate(x, func)
                X_list.append(x.copy().flatten())
                f_list.append(f)

        X = np.array(X_list)
        f = np.array(f_list)

        # Phase 2: BO loop
        q = 3
        no_improve = 0
        patience = 7
        bo_step = 0
        total_bo_steps = max(self.budget - self.n_evals, 1)

        while self.n_evals < self.budget:
            bo_step += 1
            progress = bo_step / total_bo_steps
            q_step = min(q, self.budget - self.n_evals)
            if q_step <= 0:
                break

            # Exploration restart when stuck
            if no_improve >= patience:
                candidates = []
                for _ in range(1000):
                    x_cand = np.random.uniform(lb, ub)
                    min_d = np.min(np.linalg.norm(X - x_cand, axis=1))
                    candidates.append((min_d, x_cand))
                candidates.sort(key=lambda t: -t[0])

                selected = []
                for dist, x_cand in candidates:
                    if len(selected) == 0:
                        selected.append(x_cand)
                    else:
                        if all(np.linalg.norm(x_cand - s) > 0.25 * np.linalg.norm(ub - lb)
                               for s in selected):
                            selected.append(x_cand)
                    if len(selected) >= q_step:
                        break

                while len(selected) < q_step:
                    selected.append(np.random.uniform(lb, ub))

                improved = False
                for x_next in selected[:q_step]:
                    f_next = self._evaluate(x_next, func)
                    X = np.vstack([X, x_next.reshape(1, -1)])
                    f = np.append(f, f_next)
                    if f_next < self._f_opt:
                        improved = True
                no_improve = 0 if improved else 1
                continue

            # Build and fit GP
            X_t = torch.tensor(X, dtype=torch.double)
            f_t = torch.tensor(f, dtype=torch.double).unsqueeze(-1)
            bounds_t = torch.tensor([lb, ub], dtype=torch.double)  # shape (2, dim)

            model = None
            use_original_space = True

            try:
                model = SingleTaskGP(
                    X_t, f_t,
                    input_transform=Normalize(d=self.dim, bounds=bounds_t),
                    outcome_transform=Standardize(m=1),
                )
                model.covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=self.dim),
                    outputscale_prior=torch.distributions.LogNormal(-1.0, 1.0),
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                # With Normalize transform, acq operates in original space
                acq_bounds = bounds_t
                use_original_space = True
            except Exception:
                model = None

            if model is None:
                # Fallback: manual standardization
                try:
                    X_norm = (X - lb) / (ub - lb)
                    f_mean, f_std = f.mean(), max(f.std(), 1e-10)
                    X_t2 = torch.tensor(X_norm, dtype=torch.double)
                    f_t2 = torch.tensor((f - f_mean) / f_std, dtype=torch.double).unsqueeze(-1)
                    model = SingleTaskGP(X_t2, f_t2)
                    model.covar_module = ScaleKernel(
                        MaternKernel(nu=2.5, ard_num_dims=self.dim)
                    )
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    fit_gpytorch_mll(mll)
                    acq_bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.double)
                    use_original_space = False
                except Exception:
                    # Last resort: random
                    for _ in range(q_step):
                        x_next = np.random.uniform(lb, ub)
                        f_next = self._evaluate(x_next, func)
                        X = np.vstack([X, x_next.reshape(1, -1)])
                        f = np.append(f, f_next)
                        if f_next < self._f_opt:
                            no_improve = 0
                    no_improve += 1
                    continue

            # Acquisition function selection
            if progress < 0.35:
                beta = 4.0 + 6.0 * (1.0 - progress / 0.35)
                acq = qUpperConfidenceBound(model=model, beta=beta, q=q_step)
            else:
                try:
                    outcome_transform = model.outcome_transform
                    if outcome_transform is not None and hasattr(outcome_transform, 'forward'):
                        best_f_val = outcome_transform(f_t).min().item()
                    else:
                        best_f_val = f_t.min().item()
                except Exception:
                    best_f_val = f_t.min().item()
                acq = qLogExpectedImprovement(
                    model=model, best_f=best_f_val, q=q_step,
                    MonteCarlo=True, num_samples=256
                )

            # Multi-restart acquisition maximization
            n_restarts = max(12, 2 * self.dim)
            raw_samples = 256 if q_step > 1 else 512

            try:
                best_candidate, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=acq_bounds,
                    q=q_step,
                    num_restarts=n_restarts,
                    raw_samples=raw_samples,
                    options={"maxiter": 300},
                )
            except Exception:
                best_candidate, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=acq_bounds,
                    q=q_step,
                    num_restarts=5,
                    raw_samples=128,
                )

            x_next_arr = best_candidate.detach().cpu().numpy()

            if not use_original_space:
                x_next_arr = x_next_arr * (ub - lb) + lb

            x_next_arr = np.clip(x_next_arr, lb, ub)

            improved = False
            for i in range(q_step):
                x_next = x_next_arr[i].copy()
                if len(X) > 0 and np.any(np.all(np.abs(X - x_next) < 1e-7, axis=1)):
                    x_next = x_next + np.random.normal(0, 0.01 * (ub - lb), size=self.dim)
                    x_next = np.clip(x_next, lb, ub)

                f_next = self._evaluate(x_next, func)
                X = np.vstack([X, x_next.reshape(1, -1)])
                f = np.append(f, f_next)
                if f_next < self._f_opt:
                    improved = True

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        return self._f_opt, self._x_opt