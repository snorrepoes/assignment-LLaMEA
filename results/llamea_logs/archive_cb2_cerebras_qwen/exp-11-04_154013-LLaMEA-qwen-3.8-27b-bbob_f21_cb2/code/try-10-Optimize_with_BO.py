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

    def _generate_sobol(self, lb_arr, ub_arr, n, offset=0):
        sampler = qmc.Sobol(d=self.dim, seed=self.random_seed + offset, scramble=True)
        pts = sampler.random(n).astype(np.float64)
        pts = pts * (ub_arr - lb_arr) + lb_arr
        return np.clip(pts, lb_arr, ub_arr)

    def _fit_gp(self, X, F, seed_offset=0):
        best_model = None
        best_val = -np.inf
        for attempt in range(4):
            try:
                torch.manual_seed(self.random_seed + seed_offset + attempt * 7919)
                model = SingleTaskGP(
                    X, F,
                    input_transform=InputStandardize(self.dim),
                    outcome_transform=Standardize(m=1),
                    covar_module=ScaleKernel(
                        MaternKernel(nu=2.5, ard_num_dims=self.dim)
                    ),
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll, iterations=300)
                model.eval()
                with torch.no_grad():
                    val = mll.model.log_marginal_likelihood()
                if val is not None and torch.isfinite(val) and val.item() > best_val:
                    best_val = val.item()
                    best_model = model
            except Exception:
                continue
        if best_model is None:
            model = SingleTaskGP(
                X, F,
                input_transform=InputStandardize(self.dim),
                covar_module=ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=self.dim)
                ),
            )
            model.eval()
            best_model = model
        return best_model

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_default_dtype(torch.float64)

        lb_arr = np.array(func.bounds.lb, dtype=np.float64)
        ub_arr = np.array(func.bounds.ub, dtype=np.float64)
        lb = torch.tensor(lb_arr)
        ub = torch.tensor(ub_arr)
        bounds = torch.stack([lb, ub])

        # Phase 1: Initial Design
        if self.initial_points is not None and len(self.initial_points) > 0:
            initial_pts = np.array(self.initial_points, dtype=np.float64)
            initial_pts = np.clip(initial_pts, lb_arr, ub_arr)
        else:
            n_init = min(15, self.budget)
            initial_pts = self._generate_sobol(lb_arr, ub_arr, n_init)

        for x in initial_pts:
            if self.n_calls >= self.budget:
                raise RuntimeError(f"Budget exceeded: {self.n_calls} >= {self.budget}")
            f_val = float(func(x))
            self.n_calls += 1
            self.X_obs.append(x.astype(np.float64))
            self.f_obs.append(f_val)
            if f_val < self.f_best:
                self.f_best = f_val
                self.x_best = x.copy()

        # Phase 2: Bayesian Optimization Loop
        patience = 4
        no_improve = 0
        n_bo = 0

        while self.n_calls < self.budget:
            n_bo += 1
            remaining = self.budget - self.n_calls

            # Adaptive batch size
            n_data = len(self.X_obs)
            if remaining >= 3 and n_data < 30:
                q = 3
            elif remaining >= 2:
                q = 2
            else:
                q = 1

            X = torch.tensor(np.array(self.X_obs), dtype=torch.float64)
            F = torch.tensor(np.array(self.f_obs), dtype=torch.float64).unsqueeze(-1)

            model = self._fit_gp(X, F, seed_offset=n_bo)
            f_min = F.min().item()

            # Adaptive acquisition: exploration early, exploitation later,
            # with patience-based fallback to aggressive exploration
            if no_improve >= patience:
                beta = 4.0 + 0.5 * (no_improve - patience)
                acqf = qUpperConfidenceBound(model=model, beta=beta, q=q)
            elif n_bo <= 4:
                beta = 2.5
                acqf = qUpperConfidenceBound(model=model, beta=beta, q=q)
            else:
                acqf = qLogExpectedImprovement(
                    model=model,
                    best_f=torch.tensor(f_min, dtype=torch.float64),
                    q=q,
                )

            # Optimize acquisition function
            try:
                candidates, _ = optimize_acqf(
                    acqf,
                    bounds=bounds,
                    q=q,
                    num_restarts=10,
                    raw_samples=1024,
                    options={"batch_initial_candidates": 10, "maxiter": 200},
                )
            except Exception:
                candidates = torch.tensor(
                    self._generate_sobol(lb_arr, ub_arr, q, offset=n_bo * 7),
                    dtype=torch.float64,
                )

            # Evaluate candidates
            improved = False
            for j in range(min(q, candidates.shape[0])):
                if self.n_calls >= self.budget:
                    break
                x_new = np.clip(candidates[j].numpy(), lb_arr, ub_arr)
                f_new = float(func(x_new))
                self.n_calls += 1
                self.X_obs.append(x_new)
                self.f_obs.append(f_new)
                if f_new < self.f_best - 1e-12:
                    self.f_best = f_new
                    self.x_best = x_new.copy()
                    improved = True

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

        return self.f_best, self.x_best