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
        x = np.array(x, dtype=np.float64)
        f = float(func(x))
        self.X.append(x.copy())
        self.y.append(f)
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _far_random(self, lb, ub):
        """Random point far from current best (normalized distance > 0.5)."""
        rng = np.random.RandomState(self.random_seed * 7919 + self.n_calls * 31 + 13)
        best = self.best_x
        for _ in range(500):
            x = rng.uniform(lb, ub)
            if best is None:
                return x
            d = np.linalg.norm((x - best) / (ub - lb))
            if d > 0.5:
                return x
        return rng.uniform(lb, ub)

    def _sobol_batch(self, n, lb, ub):
        from scipy.stats import qmc
        seed = (self.random_seed + self.n_calls) % (2**31 - 1)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=seed)
        pt = sampler.random(n)
        return qmc.scale(pt, lb, ub)

    def _fit_gp(self, dim):
        """Fit GP with multiple restarts; return best model by MLL."""
        Xt = torch.tensor(np.array(self.X), dtype=torch.float)
        yt = torch.tensor(np.array(self.y), dtype=torch.float).unsqueeze(-1)
        best_model, best_mll = None, -np.inf
        for r in range(3):
            try:
                torch.manual_seed(self.random_seed + r * 104729 + self.n_calls)
                model = SingleTaskGP(
                    X=Xt, y=yt,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                )
                model.covar_module.base_kernel.lengthscale = torch.full((dim,), 0.45)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_gpytorch_mll(model, max_iter=130)
                m = model.mll().item()
                if np.isfinite(m) and m > best_mll:
                    best_mll, best_model = m, model
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

        # Evaluate initial DoE
        for x in self.initial_points:
            self._evaluate(func, x)
        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        bo_iter = 0
        while self.n_calls < self.budget:
            remaining = self.budget - self.n_calls
            n_pts = len(self.X)
            progress = self.n_calls / self.budget

            # Adaptive batch size
            if remaining >= 5:
                q = 3
            elif remaining >= 2:
                q = 2
            else:
                q = 1

            # Early structured exploration if too few points
            if n_pts < dim + 2:
                n_need = min(q, remaining)
                Xs = self._sobol_batch(n_need, lb, ub)
                for x in Xs:
                    self._evaluate(func, x)
                bo_iter += 1
                continue

            # Fit surrogate
            model = self._fit_gp(dim)
            if model is None:
                Xs = self._sobol_batch(q, lb, ub)
                for x in Xs:
                    self._evaluate(func, x)
                bo_iter += 1
                continue

            # Periodic restart: inject 1 far-random point every 3 BO iters
            do_restart = (bo_iter % 3 == 2) and q >= 2 and progress < 0.85
            x_rand = self._far_random(lb, ub) if do_restart else None
            q_acq = q - 1 if do_restart else q

            # Acquisition selection
            best_f_t = torch.tensor([self.best_f], dtype=torch.float)
            if progress < 0.75:
                # UCB: beta 5 → 1.5 over first 75%
                beta = 5.0 - 3.5 * (progress / 0.75)
                acq = qUpperConfidenceBound(model=model, beta=max(beta, 1.5), q=q_acq)
            else:
                acq = qLogExpectedImprovement(model=model, best_f=best_f_t, q=q_acq)

            # Multi-start acquisition optimization
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cand = optimize_acqf(
                        f=acq, bounds=bounds, q=q_acq,
                        num_restarts=12, raw_samples=256,
                    )
            except Exception:
                cand = None

            # Assemble batch
            batch = []
            if x_rand is not None:
                batch.append(x_rand)
            if cand is not None:
                cn = cand.detach().numpy()
                for i in range(cn.shape[0]):
                    batch.append(cn[i])
            while len(batch) < q:
                batch.append(self._far_random(lb, ub))
            batch = np.clip(np.array(batch[:q], dtype=np.float64), lb, ub)

            for x in batch:
                self._evaluate(func, x)
            bo_iter += 1

        return self.best_f, self.best_x