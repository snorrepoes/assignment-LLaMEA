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

    def _fit_gp(self):
        X_tensor = torch.tensor(np.array(self.X), dtype=torch.float)
        y_tensor = torch.tensor(np.array(self.y), dtype=torch.float).unsqueeze(-1)

        model = SingleTaskGP(
            X=X_tensor,
            y=y_tensor,
            input_transform=Normalize(d=self.dim),
            outcome_transform=Standardize(m=1),
        )
        model.covar_module.base_kernel.lengthscale = torch.full((self.dim,), 0.08)
        model.likelihood.noise = 0.005

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(model, progress_bar=False)
        return model

    def _diverse_point(self, lb, ub):
        if len(self.X) == 0:
            return np.random.uniform(lb, ub)
        X_arr = np.array(self.X)
        rng = (ub - lb)
        X_norm = (X_arr - lb) / rng
        best_dist = -1.0
        best_point = np.random.uniform(lb, ub)
        for _ in range(100):
            cand = np.random.uniform(lb, ub)
            cand_norm = (cand - lb) / rng
            dists = np.linalg.norm(X_norm - cand_norm, axis=1)
            min_d = dists.min()
            if min_d > best_dist:
                best_dist = min_d
                best_point = cand
        return best_point

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        dim = self.dim
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)

        # Phase 1: Evaluate initial points
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # Trust region parameters
        domain_range = ub - lb
        tr_radius_init = 0.12 * domain_range
        tr_radius_min = 0.05 * domain_range
        patience = 4

        tr_center = self.best_x.copy()
        tr_radius = tr_radius_init.copy()
        tr_failures = 0

        remaining = self.budget - self.n_calls
        step = 0

        while self.n_calls < self.budget:
            step += 1

            # Compute effective TR bounds
            tr_lb = np.maximum(tr_center - tr_radius, lb)
            tr_ub = np.minimum(tr_center + tr_radius, ub)

            # Degenerate TR: relocate with diverse point
            if np.all(tr_ub - tr_lb < tr_radius_min * 0.4):
                tr_center = self._diverse_point(lb, ub)
                tr_radius = tr_radius_init.copy()
                tr_failures = 0
                self._evaluate(func, tr_center)
                if self.n_calls >= self.budget:
                    break
                tr_center = self.best_x.copy()
                continue

            # Fit GP
            try:
                model = self._fit_gp()
                gp_ok = True
            except Exception:
                gp_ok = False

            if not gp_ok:
                x_new = np.random.uniform(tr_lb, tr_ub)
                old_best = self.best_f
                self._evaluate(func, x_new)
                if self.best_f < old_best - 1e-10:
                    tr_center = self.best_x.copy()
                    tr_radius = np.maximum(tr_radius * 0.9, tr_radius_min)
                    tr_failures = 0
                else:
                    tr_failures += 1
                    if tr_failures >= patience:
                        tr_center = self._diverse_point(lb, ub)
                        tr_radius = tr_radius_init.copy()
                        tr_failures = 0
                        if self.n_calls < self.budget:
                            self._evaluate(func, tr_center)
                            if self.n_calls >= self.budget:
                                break
                            tr_center = self.best_x.copy()
                continue

            best_f_tensor = torch.tensor([self.best_f], dtype=torch.float)
            progress = step / max(remaining, 1)

            # Adaptive acquisition: UCB for exploration, EI for exploitation
            if progress < 0.3:
                acq_f = qUpperConfidenceBound(model=model, beta=5.0, q=1)
            elif progress < 0.65:
                acq_f = qUpperConfidenceBound(model=model, beta=2.0, q=1)
            else:
                acq_f = qLogExpectedImprovement(model=model, best_f=best_f_tensor, q=1)

            # Optimize within TR
            tr_lb_t = torch.tensor(tr_lb, dtype=torch.float)
            tr_ub_t = torch.tensor(tr_ub, dtype=torch.float)
            for d in range(dim):
                if tr_ub_t[d] - tr_lb_t[d] < 1e-4:
                    mid = (tr_ub_t[d] + tr_lb_t[d]) / 2
                    tr_lb_t[d] = mid - 5e-3
                    tr_ub_t[d] = mid + 5e-3

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=[tr_lb_t, tr_ub_t],
                        q=1,
                        num_restarts=25,
                        raw_samples=256,
                    )
            except Exception:
                x_new = np.random.uniform(tr_lb, tr_ub)
                old_best = self.best_f
                self._evaluate(func, x_new)
                if self.best_f < old_best - 1e-10:
                    tr_center = self.best_x.copy()
                    tr_radius = np.maximum(tr_radius * 0.9, tr_radius_min)
                    tr_failures = 0
                else:
                    tr_failures += 1
                    if tr_failures >= patience:
                        tr_center = self._diverse_point(lb, ub)
                        tr_radius = tr_radius_init.copy()
                        tr_failures = 0
                        if self.n_calls < self.budget:
                            self._evaluate(func, tr_center)
                            if self.n_calls >= self.budget:
                                break
                            tr_center = self.best_x.copy()
                continue

            x_new = candidates.detach().cpu().numpy().flatten()
            x_new = np.clip(x_new, tr_lb, tr_ub)

            old_best = self.best_f
            self._evaluate(func, x_new)

            if self.best_f < old_best - 1e-10:
                tr_center = self.best_x.copy()
                tr_radius = np.maximum(tr_radius * 0.88, tr_radius_min)
                tr_failures = 0
            else:
                tr_failures += 1
                if tr_failures >= patience:
                    tr_center = self._diverse_point(lb, ub)
                    tr_radius = tr_radius_init.copy()
                    tr_failures = 0
                    if self.n_calls < self.budget:
                        self._evaluate(func, tr_center)
                        if self.n_calls >= self.budget:
                            break
                        tr_center = self.best_x.copy()

        return self.best_f, self.best_x