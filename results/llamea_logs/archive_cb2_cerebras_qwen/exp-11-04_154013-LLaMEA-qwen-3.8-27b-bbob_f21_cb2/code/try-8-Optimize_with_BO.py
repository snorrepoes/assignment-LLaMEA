import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

# Try to import Standardize from different locations depending on BoTorch version
try:
    from botorch.models.transforms.input import Standardize
except ImportError:
    try:
        from botorch.models.transforms import Standardize
    except ImportError:
        Standardize = None


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = np.array(initial_points, dtype=np.float64)
        self.random_seed = random_seed
        self.n_calls = 0
        self.max_calls = budget

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        dim = self.dim

        X_list = []
        f_list = []

        # Evaluate initial points
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

        q = 3  # batch size
        bounds = torch.stack([
            torch.tensor(lb, dtype=torch.float32),
            torch.tensor(ub, dtype=torch.float32),
        ])

        n_initial = len(self.initial_points)
        # Switch from exploration (UCB) to exploitation (EI) after 40% of remaining budget
        total_bo_iterations = (self.max_calls - n_initial) // q
        switch_iter = max(1, int(0.4 * total_bo_iterations))

        for iteration in range(total_bo_iterations):
            if self.n_calls >= self.max_calls:
                break
            current_q = min(q, self.max_calls - self.n_calls)
            if current_q <= 0:
                break

            X_t = torch.tensor(X_all, dtype=torch.float32)
            f_t = torch.tensor(f_all, dtype=torch.float32).unsqueeze(-1)

            # Fit GP surrogate
            gp_fitted = False
            model = None
            for attempt in range(3):
                try:
                    input_transform = None
                    if Standardize is not None:
                        input_transform = Standardize(dim)
                    model = SingleTaskGP(
                        train_X=X_t,
                        train_Y=f_t,
                        input_transform=input_transform,
                        covar_module=ScaleKernel(
                            batch_shape=torch.Size([1]),
                            outputscale_prior=torch.distributions.LogNormal(-1, 1),
                        ),
                    )
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    fit_gpytorch_mll(mll, options={"maxiter": 300})
                    gp_fitted = True
                    break
                except Exception:
                    # Try without input transform
                    try:
                        model = SingleTaskGP(
                            train_X=X_t,
                            train_Y=f_t,
                        )
                        mll = ExactMarginalLogLikelihood(model.likelihood, model)
                        fit_gpytorch_mll(mll, options={"maxiter": 200})
                        gp_fitted = True
                        break
                    except Exception:
                        continue

            if not gp_fitted:
                # Random search fallback
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

            # Choose acquisition function based on phase
            if iteration < switch_iter:
                # Exploration phase: UCB
                beta = 10.0 * np.sqrt(np.log(max(1, self.n_calls)))
                acqfn = qUpperConfidenceBound(
                    model=model,
                    beta=beta,
                    q=current_q,
                )
            else:
                # Exploitation phase: EI
                acqfn = qLogExpectedImprovement(
                    model=model,
                    best_f=torch.tensor([best_f], dtype=torch.float32),
                    q=current_q,
                    num_samples=256,
                )

            # Optimize acquisition with multi-restart
            try:
                candidates, _ = optimize_acqf(
                    acq_function=acqfn,
                    bounds=bounds,
                    q=current_q,
                    num_restarts=12,
                    raw_samples=2048,
                    options={"maxiter": 150},
                )
            except Exception:
                # Fallback: random candidates
                candidates = torch.rand(current_q, dim, dtype=torch.float32) * (
                    torch.tensor(ub, dtype=torch.float32) - torch.tensor(lb, dtype=torch.float32)
                ) + torch.tensor(lb, dtype=torch.float32)

            # Evaluate candidates
            for i in range(current_q):
                if self.n_calls >= self.max_calls:
                    break
                x_new = candidates[i].detach().numpy().astype(np.float64)
                # Clamp to bounds
                x_new = np.clip(x_new, lb, ub)
                f_new = float(func(x_new))
                X_all = np.vstack([X_all, x_new])
                f_all = np.concatenate([f_all, [f_new]])
                self.n_calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

        return best_f, best_x