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

    def _fit_gp(self, X_norm, f):
        """Fit GP model, return model or None on failure."""
        f_mean = f.mean()
        f_std = f.std()
        if f_std < 1e-10:
            f_z = f - f_mean
        else:
            f_z = (f - f_mean) / f_std

        X_t = torch.tensor(X_norm, dtype=torch.double)
        f_t = torch.tensor(f_z, dtype=torch.double).unsqueeze(-1)

        try:
            model = SingleTaskGP(X_t, f_t, noise_level=1e-8)
            model.covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=self.dim)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            return model, X_t, f_t
        except Exception:
            try:
                model = SingleTaskGP(X_t, f_t, noise_level=1e-6)
                model.covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=self.dim)
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll, max_iter=30)
                return model, X_t, f_t
            except Exception:
                return None, None, None

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        scale = ub - lb

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
                X_list.append(x.copy())
                f_list.append(f)

        # Complement with Latin Hypercube sampling for space-filling
        target_init = min(15, self.budget)
        if len(X_list) < target_init:
            n_needed = target_init - len(X_list)
            lhs = qmc.LatinHypercube(d=self.dim, seed=self.random_seed)
            points = lhs.random(n_needed)
            points = points * scale + lb
            for x in points:
                f = self._evaluate(x, func)
                X_list.append(x.copy())
                f_list.append(f)

        X = np.array(X_list)
        f = np.array(f_list)

        # Phase 2: Bayesian Optimization loop
        no_improve = 0
        patience = 7
        batch_size = 3
        remaining_budget = self.budget - self.n_evals
        total_bo_steps = max((remaining_budget + batch_size - 1) // batch_size, 1)
        bo_step = 0

        while self.n_evals < self.budget:
            bo_step += 1
            remaining = self.budget - self.n_evals
            q = min(batch_size, remaining)
            progress = (bo_step - 1) / max(total_bo_steps, 1)

            if no_improve >= patience:
                # Exploration restart: pick q diverse far points
                X_norms = (X - lb) / scale
                candidates = []
                n_candidates = 600
                rng = np.random.RandomState(self.random_seed + bo_step)
                for _ in range(n_candidates):
                    x_cand = rng.uniform(lb, ub, size=self.dim)
                    x_cand_norm = (x_cand - lb) / scale
                    min_d = np.min(np.linalg.norm(X_norms - x_cand_norm, axis=1))
                    candidates.append((min_d, x_cand))
                candidates.sort(key=lambda t: -t[0])

                selected = []
                min_diversity = 0.4 * np.linalg.norm(scale)
                for dist, x_cand in candidates:
                    if len(selected) >= q:
                        break
                    diverse = True
                    for s in selected:
                        if np.linalg.norm(x_cand - s) < min_diversity:
                            diverse = False
                            break
                    if diverse:
                        selected.append(x_cand)

                while len(selected) < q:
                    selected.append(rng.uniform(lb, ub))

                improved = False
                for x_next in selected[:q]:
                    f_next = self._evaluate(x_next, func)
                    if f_next < self._f_opt - 1e-10:
                        improved = True
                    X = np.vstack([X, x_next.reshape(1, -1)])
                    f = np.append(f, f_next)

                if improved:
                    no_improve = 0
                else:
                    no_improve = 0  # Reset to avoid immediate re-trigger
                continue

            # Build GP model
            X_norm = (X - lb) / scale
            result = self._fit_gp(X_norm, f)
            model = result[0]

            if model is None:
                # GP fitting failed: use random exploration
                rng = np.random.RandomState(self.random_seed + 1000 * bo_step)
                for _ in range(q):
                    x_next = rng.uniform(lb, ub)
                    f_next = self._evaluate(x_next, func)
                    if f_next < self._f_opt - 1e-10:
                        no_improve = 0
                    else:
                        no_improve += 1
                    X = np.vstack([X, x_next.reshape(1, -1)])
                    f = np.append(f, f_next)
                continue

            _, _, f_t = result
            f_best = f_t.min().item()

            # Adaptive acquisition: UCB early, LogEI late
            if progress < 0.35:
                beta = 6.0 - 4.0 * (progress / 0.35)
                acq = qUpperConfidenceBound(model=model, beta=beta)
            elif progress < 0.65:
                beta = 2.0 - 1.0 * ((progress - 0.35) / 0.30)
                acq = qUpperConfidenceBound(model=model, beta=max(beta, 0.5))
            else:
                acq = qLogExpectedImprovement(model=model, best_f=f_best)

            bounds = torch.tensor(
                [[0.0] * self.dim, [1.0] * self.dim], dtype=torch.double
            )

            n_restarts = 20
            try:
                best_candidate, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=bounds,
                    q=q,
                    num_restarts=n_restarts,
                    raw_samples=256,
                )
            except Exception:
                # Fallback: use random candidates
                rng = np.random.RandomState(self.random_seed + 2000 * bo_step)
                best_candidate = torch.tensor(
                    rng.uniform(0, 1, size=(q, self.dim)), dtype=torch.double
                )

            x_next = best_candidate.detach().cpu().numpy()
            x_next = x_next * scale + lb
            x_next = np.clip(x_next, lb, ub)

            # Evaluate batch
            improved = False
            for i in range(q):
                x_i = x_next[i].copy()
                # Avoid near-duplicates
                if len(X) > 0:
                    dists = np.linalg.norm(X - x_i, axis=1)
                    if np.min(dists) < 1e-6 * np.linalg.norm(scale):
                        rng = np.random.RandomState(self.random_seed + 3000 * bo_step + i)
                        x_i = x_i + rng.normal(0, 0.02 * np.linalg.norm(scale), size=self.dim)
                        x_i = np.clip(x_i, lb, ub)

                f_next = self._evaluate(x_i, func)
                if f_next < self._f_opt - 1e-10:
                    improved = True
                X = np.vstack([X, x_i.reshape(1, -1)])
                f = np.append(f, f_next)

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        return self._f_opt, self._x_opt