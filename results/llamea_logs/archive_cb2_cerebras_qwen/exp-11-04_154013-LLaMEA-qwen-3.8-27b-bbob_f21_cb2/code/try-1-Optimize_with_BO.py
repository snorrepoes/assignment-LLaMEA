import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = np.array(initial_points, dtype=np.float64)
        self.random_seed = random_seed
        self.n_calls = 0
        self.max_calls = budget

    def __call__(self, func):
        # Set all random seeds
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        dim = self.dim

        X_list = []
        f_list = []

        # ---- Evaluate initial points (count against budget) ----
        for x in self.initial_points:
            if self.n_calls >= self.max_calls:
                raise Exception("Budget exceeded: initial points exceed budget")
            f_val = func(np.array(x, dtype=np.float64))
            X_list.append(np.array(x, dtype=np.float64))
            f_list.append(float(f_val))
            self.n_calls += 1

        X_all = np.array(X_list)
        f_all = np.array(f_list)

        best_idx = np.argmin(f_all)
        best_f = float(f_all[best_idx])
        best_x = X_all[best_idx].copy()

        # ---- Bayesian Optimization loop ----
        q = 4  # batch size per iteration
        bounds = torch.stack([
            torch.tensor(lb, dtype=torch.float32),
            torch.tensor(ub, dtype=torch.float32),
        ])

        while self.n_calls < self.max_calls:
            current_q = min(q, self.max_calls - self.n_calls)
            if current_q <= 0:
                break

            X_t = torch.tensor(X_all, dtype=torch.float32)
            f_t = torch.tensor(f_all, dtype=torch.float32).unsqueeze(-1)

            # --- Fit GP surrogate ---
            try:
                model = SingleTaskGP(
                    train_X=X_t,
                    train_Y=f_t,
                    input_transform=Standardize(dim),
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll, options={"maxiter": 500})
            except Exception:
                # Fallback: random evaluations
                for _ in range(current_q):
                    if self.n_calls >= self.max_calls:
                        break
                    x_new = np.random.uniform(lb, ub)
                    f_new = float(func(x_new))
                    X_all = np.vstack([X_all, x_new])
                    f_all = np.concatenate([f_all, [f_new]])
                    self.n_calls += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                continue

            # --- Acquisition: qLogExpectedImprovement ---
            acqfn = qLogExpectedImprovement(
                model=model,
                best_f=torch.tensor([best_f], dtype=torch.float32),
                num_samples=256,
            )

            # --- Optimize acquisition (L-CMA-ES with restarts) ---
            candidates, _ = optimize_acqf(
                acq_function=acqfn,
                bounds=bounds,
                q=current_q,
                num_restarts=10,
                raw_samples=2048,
                options={"maxiter": 200},
            )

            # --- Evaluate candidates and update record ---
            for i in range(current_q):
                if self.n_calls >= self.max_calls:
                    break
                x_new = candidates[i].detach().numpy().astype(np.float64)
                f_new = float(func(x_new))
                X_all = np.vstack([X_all, x_new])
                f_all = np.concatenate([f_all, [f_new]])
                self.n_calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

        return best_f, best_x