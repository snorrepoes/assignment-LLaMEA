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
            raise Exception(
                f"Budget exceeded: call {self.n_calls} > budget {self.budget}"
            )
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
        bounds = torch.stack([
            torch.tensor(lb, dtype=torch.float32),
            torch.tensor(ub, dtype=torch.float32),
        ])

        # ---- Phase 1: Evaluate initial points (count against budget) ----
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # ---- Phase 2: Bayesian Optimization loop ----
        patience_limit = 7
        patience_counter = 0
        gp_fail_count = 0

        while self.n_calls < self.budget:
            n_points = len(self.X)
            progress = self.n_calls / self.budget

            # Not enough data for a meaningful GP -> random
            if n_points < dim + 3:
                x_new = np.random.uniform(lb, ub)
                f_new = self._evaluate(func, x_new)
                if f_new < self.best_f:
                    patience_counter = 0
                else:
                    patience_counter += 1
                continue

            # Patience: stuck for too long -> random exploration
            if patience_counter >= patience_limit:
                x_new = np.random.uniform(lb, ub)
                f_new = self._evaluate(func, x_new)
                if f_new < self.best_f:
                    patience_counter = 0
                else:
                    patience_counter += 1
                continue

            # Stochastic diversity: with some probability, explore randomly
            # (helps escape local basins in multimodal landscapes)
            diversity_prob = 0.15 * max(0.0, 1.0 - progress) + 0.05
            if n_points > 2 * dim and np.random.random() < diversity_prob:
                x_new = np.random.uniform(lb, ub)
                f_new = self._evaluate(func, x_new)
                if f_new < self.best_f:
                    patience_counter = 0
                else:
                    patience_counter += 1
                continue

            # ---- Fit GP (negate y to align with BoTorch maximization convention) ----
            X_tensor = torch.tensor(np.array(self.X), dtype=torch.float32)
            # Negate: GP models -f, so maximizing acquisition = minimizing f
            y_tensor = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1) * (-1.0)

            model = None
            try:
                model = SingleTaskGP(
                    X=X_tensor,
                    y=y_tensor,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                )
                # Short lengthscales to capture local multimodal structure
                model.covar_module.base_kernel.lengthscale = torch.full((dim,), 0.25)
                model.covar_module.base_kernel.output_covar = (
                    model.covar_module.base_kernel.output_covar
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_gpytorch_mll(model)
                gp_fail_count = 0
            except Exception:
                gp_fail_count += 1
                if gp_fail_count >= 3:
                    # GP keeps failing -> random exploration
                    x_new = np.random.uniform(lb, ub)
                    f_new = self._evaluate(func, x_new)
                    if f_new < self.best_f:
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    gp_fail_count = 0
                else:
                    x_new = np.random.uniform(lb, ub)
                    f_new = self._evaluate(func, x_new)
                    if f_new < self.best_f:
                        patience_counter = 0
                    else:
                        patience_counter += 1
                continue

            # ---- Acquisition function selection ----
            # Best f in maximization sense (negated)
            best_f_max = torch.tensor([-self.best_f], dtype=torch.float32)

            if progress < 0.45:
                # Exploration phase: UCB with decaying beta (aggressive early)
                beta = 8.0 * (1.0 - progress / 0.45) + 2.0
                acq_f = qUpperConfidenceBound(model=model, beta=beta, q=1)
            else:
                # Exploitation phase: Expected Improvement
                acq_f = qLogExpectedImprovement(
                    model=model, best_f=best_f_max, q=1
                )

            # ---- Multi-start acquisition optimization ----
            candidates = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=bounds,
                        q=1,
                        num_restarts=25,
                        raw_samples=512,
                        batch_initial_points=25,
                    )
            except Exception:
                # Fallback: random
                x_new = np.random.uniform(lb, ub)
                f_new = self._evaluate(func, x_new)
                if f_new < self.best_f:
                    patience_counter = 0
                else:
                    patience_counter += 1
                continue

            if candidates is None:
                x_new = np.random.uniform(lb, ub)
                f_new = self._evaluate(func, x_new)
                if f_new < self.best_f:
                    patience_counter = 0
                else:
                    patience_counter += 1
                continue

            # ---- Evaluate the selected point ----
            x_new = candidates.detach().numpy().flatten()
            f_new = self._evaluate(func, x_new)
            if f_new < self.best_f:
                patience_counter = 0
            else:
                patience_counter += 1

        return self.best_f, self.best_x