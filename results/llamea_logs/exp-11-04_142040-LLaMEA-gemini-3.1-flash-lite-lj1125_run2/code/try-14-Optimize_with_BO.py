import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior

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
        for x in self.initial_points:
            if len(train_x) >= self.budget: break
            train_x.append(torch.tensor(x, dtype=torch.double))
            train_y.append(func(x))

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Use log-transformed Y to compress the plateau and emphasize the well
            # Shift Y to be positive if necessary
            Y_trans = -torch.log(Y - Y.min() + 1e-12)
            
            # Use a Matern kernel with automatic lengthscale discovery
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_trans, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.no_grad():
                fit_gpytorch_mll(mll)
            
            # TuRBO-like heuristic: limit search space to a trust region around the best point
            best_idx = torch.argmin(Y.view(-1))
            best_x = X[best_idx]
            tr_lb = torch.clamp(best_x - 1.5, lb, ub)
            tr_ub = torch.clamp(best_x + 1.5, lb, ub)
            
            qNEI = qNoisyExpectedImprovement(model, X_baseline=X)
            new_x, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=15,
                raw_samples=512,
            )
            
            new_x = new_x.detach().squeeze()
            new_f = func(new_x.numpy())
            
            train_x.append(new_x)
            train_y.append(new_f)

        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()