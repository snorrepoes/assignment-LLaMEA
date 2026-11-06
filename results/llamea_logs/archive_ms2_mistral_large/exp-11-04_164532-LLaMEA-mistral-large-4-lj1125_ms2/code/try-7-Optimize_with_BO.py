import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        self.random_seed = random_seed

    def __call__(self, func):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.set_default_dtype(torch.float64)

        if self.initial_points.shape[0] > self.budget:
            raise ValueError("Number of initial points exceeds budget")

        lb = torch.tensor(func.bounds.lb, dtype=torch.float64)
        ub = torch.tensor(func.bounds.ub, dtype=torch.float64)
        bounds = torch.stack([lb, ub])

        count = 0

        def evaluate(x):
            nonlocal count
            if count >= self.budget:
                raise RuntimeError("Function call budget exceeded")
            count += 1
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy().reshape(-1)
            else:
                x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            return float(func(x_np))

        X_list = []
        Y_list = []
        for x in self.initial_points:
            X_list.append(x)
            Y_list.append(evaluate(x))

        X = torch.tensor(np.array(X_list), dtype=torch.float64)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float64).unsqueeze(-1)

        def transform_y(y):
            return torch.sign(y) * torch.log1p(torch.abs(y))

        while count < self.budget:
            Y_trans = transform_y(Y)
            try:
                model = SingleTaskGP(X, Y_trans, outcome_transform=Standardize(m=1))
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                best_f = Y.min().item()
                best_f_trans = float(transform_y(torch.tensor([[best_f]], dtype=torch.float64)).item())
                acq = qLogExpectedImprovement(model, best_f=best_f_trans)

                candidate, _ = optimize_acqf(
                    acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                    options={"maxiter": 50},
                )
                if candidate.dim() == 3:
                    x_new = candidate[0, 0]
                else:
                    x_new = candidate[0]
            except Exception:
                x_new = lb + (ub - lb) * torch.rand(self.dim, dtype=torch.float64)

            y_new = evaluate(x_new)
            X = torch.cat([X, x_new.reshape(1, -1)], dim=0)
            Y = torch.cat([Y, torch.tensor([[y_new]], dtype=torch.float64)], dim=0)

        best_idx = torch.argmin(Y).item()
        best_f = Y[best_idx].item()
        best_x = X[best_idx].cpu().numpy()
        return best_f, best_x