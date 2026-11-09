import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.transforms import Standardize

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.double)
        train_x = []
        train_y = []

        # Evaluate initial points
        for x in self.initial_points:
            f = func(x)
            train_x.append(torch.tensor(x, dtype=torch.double))
            train_y.append(f)

        if len(train_x) > self.budget:
            raise ValueError("Initial points exceed budget.")

        # BO loop
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Log transform and Standardize to handle extreme ranges and flat plateaus
            Y_log = torch.log(Y - Y.min() + 1e-9)
            model = SingleTaskGP(X, Y_log, outcome_transform=Standardize(m=1))
            model.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.no_grad():
                fit_gpytorch_mll(mll)
            
            # Use Log-EI for better sensitivity in low-value regions
            acqf = qLogExpectedImprovement(model, best_f=Y_log.min())
            
            new_x, _ = optimize_acqf(
                acqf, bounds=bounds, q=1, num_restarts=20, raw_samples=512
            )
            
            new_x_np = new_x.detach().numpy()[0]
            new_f = func(new_x_np)
            
            train_x.append(new_x.squeeze(0))
            train_y.append(new_f)
            
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()