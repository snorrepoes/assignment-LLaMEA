import numpy as np
import torch
import warnings
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings('ignore')


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = np.array(initial_points, dtype=np.float64)
        self.random_seed = random_seed
        self.n_calls = 0

    def _log_transform(self, F_torch):
        """Log-transform to compress multi-scale objective for GP fitting."""
        return torch.log(F_torch.clamp(min=1e-30))

    def __call__(self, func):
        # Seed all stochastic components
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        lb_t = torch.tensor(lb, dtype=torch.float32)
        ub_t = torch.tensor(ub, dtype=torch.float32)

        # --- Phase 1: Evaluate initial DoE points ---
        X_list, F_list = [], []
        for x in self.initial_points:
            if self.n_calls >= self.budget:
                raise Exception("Evaluation budget exceeded during initial points.")
            f_val = func(np.asarray(x, dtype=np.float64))
            X_list.append(np.asarray(x, dtype=np.float64))
            F_list.append(float(f_val))
            self.n_calls += 1

        X_arr = np.array(X_list)
        F_arr = np.array(F_list)

        best_f = F_arr.min()
        best_x = X_arr[F_arr.argmin()].copy()

        remaining = self.budget - self.n_calls
        if remaining <= 0:
            return best_f, best_x

        # --- Phase 2: Bayesian Optimization loop ---
        for i in range(remaining):
            X_torch = torch.tensor(X_arr, dtype=torch.float32)
            F_torch = torch.tensor(F_arr, dtype=torch.float32).unsqueeze(-1)

            # Log-transform to handle orders-of-magnitude differences
            F_log = self._log_transform(F_torch)

            # Build and fit GP (Matern 5/2 with ARD, input-normalized)
            model = SingleTaskGP(
                X=X_torch,
                Y=F_log,
                input_transform=Normalize(d=self.dim),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll, max_iter=100)
            except Exception:
                pass  # keep default hyperparameters if fit fails

            # Adaptive beta: start with aggressive exploration, shift to exploitation
            progress = i / max(remaining - 1, 1)
            beta = 5.0 * (1.0 - progress) + 0.5 * progress

            # UCB acquisition (maximize=False for minimization)
            acqf = qUpperConfidenceBound(model=model, beta=beta, maximize=False)

            # Optimize acquisition via L-BFGS with multistart
            try:
                X_cand, _ = optimize_acqf(
                    f=acqf,
                    bounds=torch.stack([lb_t, ub_t]),
                    q=1,
                    num_restarts=10,
                    raw_samples=2048,
                )
            except Exception:
                X_cand = lb_t + (ub_t - lb_t) * torch.rand(1, self.dim)

            x_new = X_cand[0].detach().numpy().astype(np.float64)
            x_new = np.clip(x_new, lb, ub)

            # Evaluate
            self.n_calls += 1
            if self.n_calls > self.budget:
                raise Exception("Evaluation budget exceeded.")
            f_new = float(func(x_new))

            X_arr = np.vstack([X_arr, x_new.reshape(1, -1)])
            F_arr = np.append(F_arr, f_new)

            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        return best_f, best_x