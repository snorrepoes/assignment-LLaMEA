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
        train_x, train_y = [], []

        # Evaluate initial points
        for x in self.initial_points:
            train_x.append(torch.tensor(x, dtype=torch.double))
            train_y.append(float(func(x)))

        # TuRBO-1 parameters
        length_min, length_max = 0.5, 2.0
        length = 0.8
        best_val = min(train_y)
        best_x = train_x[np.argmin(train_y)]
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            # Log transform to handle plateaus: target is small values, map to large values
            Y_trans = -torch.log(Y - Y.min() + 1e-12)
            
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_trans, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define Trust Region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qNoisyExpectedImprovement(model, X_baseline=X)
            new_x, _ = optimize_acqf(acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, num_restarts=10, raw_samples=256)
            
            new_x = new_x.detach().flatten()
            new_f = float(func(new_x.numpy()))
            
            train_x.append(new_x)
            train_y.append(new_f)
            
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                length = min(length * 1.1, length_max)
            else:
                length = max(length * 0.8, length_min)

        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()