import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        train_x = []
        train_y = []
        
        # Initial points
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
            
            # Log transform Y to handle magnitude differences
            Y_transformed = torch.log(Y - Y.min() + 1e-9)
            
            model = SingleTaskGP(X, Y_transformed)
            model.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            qNEI = qNoisyExpectedImprovement(model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                qNEI, bounds=torch.stack([lb, ub]), q=1, num_restarts=10, raw_samples=100
            )
            
            new_x = new_x.detach().numpy()[0]
            new_f = func(new_x)
            
            train_x.append(torch.tensor(new_x, dtype=torch.double))
            train_y.append(new_f)
            
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()