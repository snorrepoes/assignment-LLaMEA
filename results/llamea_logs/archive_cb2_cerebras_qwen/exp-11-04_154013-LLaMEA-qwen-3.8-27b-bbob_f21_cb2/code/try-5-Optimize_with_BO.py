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

    def _far_random_point(self, lb, ub, n_candidates=100):
        """Return a random point that is far from all existing observations."""
        X_arr = np.array(self.X)
        best_dist = -1e18
        best_point = None
        for _ in range(n_candidates):
            x = np.random.uniform(lb, ub)
            dists = np.linalg.norm(X_arr - x, axis=1)
            min_dist = dists.min()
            if min_dist > best_dist:
                best_dist = min_dist
                best_point = x
        return best_point

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        dim = self.dim
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        bounds = torch.stack([torch.tensor(lb, dtype=torch.float),
                              torch.tensor(ub, dtype=torch.float)])

        # Phase 1: Evaluate initial points
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # Phase 2: Sequential BO with GP-Hedge
        patience = 5
        no_improve_count = 0
        prev_best = self.best_f

        # Hedge weights: [LogEI, UCB(beta=2), UCB(beta=8)]
        hedge_weights = np.array([1.0, 1.0, 1.0])
        eta = 1.0

        while self.n_calls < self.budget:
            n_points = len(self.X)

            # Patience-based exploration: jump to a far point
            if self.best_f < prev_best - 1e-10:
                no_improve_count = 0
                prev_best = self.best_f
            else:
                no_improve_count += 1

            if no_improve_count >= patience and n_points >= 2 * dim:
                x_new = self._far_random_point(lb, ub)
                self._evaluate(func, x_new)
                no_improve_count = 0
                prev_best = self.best_f
                hedge_weights[1] += 0.5
                hedge_weights[2] += 0.5
                continue

            if n_points < self.dim + 2:
                x_new = np.random.uniform(lb, ub)
                self._evaluate(func, x_new)
                continue

            X_tensor = torch.tensor(np.array(self.X), dtype=torch.float)
            y_tensor = torch.tensor(np.array(self.y), dtype=torch.float).unsqueeze(-1)

            # Fit GP surrogate
            try:
                model = SingleTaskGP(
                    X=X_tensor,
                    y=y_tensor,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                )
                # Shorter lengthscale to capture local structure in multimodal landscape
                model.covar_module.base_kernel.lengthscale = torch.full((dim,), 0.8)
                model.covar_module.base_kernel.lengthscale_constraint.set_bounds(0.1, 5.0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_gpytorch_mll(model, max_retries=3)
            except Exception:
                x_new = np.random.uniform(lb, ub)
                self._evaluate(func, x_new)
                continue

            best_f_tensor = torch.tensor([self.best_f], dtype=torch.float)

            # Pool of acquisition functions
            acq_pool = [
                qLogExpectedImprovement(model=model, best_f=best_f_tensor, q=1),
                qUpperConfidenceBound(model=model, beta=2.0, q=1),
                qUpperConfidenceBound(model=model, beta=8.0, q=1),
            ]

            # GP-Hedge: select acquisition with small random perturbation for diversity
            perturbed = hedge_weights + np.random.normal(0, 0.15, size=3)
            k_selected = int(np.argmax(perturbed))
            acq_f = acq_pool[k_selected]

            # Multi-restart acquisition optimization
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=bounds,
                        q=1,
                        num_restarts=20,
                        raw_samples=512,
                        batch_initial_points=10,
                    )
            except Exception:
                x_new = np.random.uniform(lb, ub)
                self._evaluate(func, x_new)
                continue

            x_new = candidates.numpy().flatten()
            prev_best = self.best_f
            f_new = self._evaluate(func, x_new)

            # Update hedge weights
            if f_new < prev_best - 1e-10:
                hedge_weights[k_selected] += eta
            else:
                hedge_weights[k_selected] -= eta * 0.4
                for j in range(3):
                    if j != k_selected:
                        hedge_weights[j] += eta * 0.1

        return self.best_f, self.best_x