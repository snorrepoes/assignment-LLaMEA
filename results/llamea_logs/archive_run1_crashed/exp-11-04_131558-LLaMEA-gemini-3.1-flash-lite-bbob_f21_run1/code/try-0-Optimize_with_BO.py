import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.noisy_ei import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        X_list = []
        Y_list = []

        # Evaluate initial points
        for x_init in self.initial_points:
            X_list.append(torch.tensor(x_init, dtype=torch.double))
            Y_list.append(func(x_init))
        
        train_X = torch.stack(X_list)
        train_Y = torch.tensor(Y_list, dtype=torch.double).unsqueeze(-1)
        
        while len(train_X) < self.budget:
            # Fit GP surrogate
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_X, train_Y, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Acquisition function
            qNEI = qNoisyExpectedImprovement(model=model, X_baseline=train_X)
            
            # Optimize acquisition
            new_x, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            # Evaluate objective
            new_x = new_x.detach().numpy().flatten()
            new_y = func(new_x)
            
            train_X = torch.cat([train_X, torch.tensor([new_x], dtype=torch.double)])
            train_Y = torch.cat([train_Y, torch.tensor([[new_y]], dtype=torch.double)])
            
        best_idx = torch.argmin(train_Y)
        return float(train_Y[best_idx]), train_X[best_idx].numpy()