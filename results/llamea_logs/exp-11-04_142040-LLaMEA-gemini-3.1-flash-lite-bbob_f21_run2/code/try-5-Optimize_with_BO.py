import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        X_list = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        Y_list = [torch.tensor([func(x)], dtype=torch.double) for x in self.initial_points]
        
        if len(X_list) > self.budget:
            raise ValueError("Initial points exceed budget.")

        best_f = float('inf')
        best_x = None
        for x, y in zip(X_list, Y_list):
            if y.item() < best_f:
                best_f = y.item()
                best_x = x.numpy()

        while len(X_list) < self.budget:
            train_X = torch.stack(X_list)
            train_Y = torch.stack(Y_list)
            
            # Using ARD kernel to capture dimension-specific sensitivities in the multimodal landscape
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_X, train_Y, mean_module=ConstantMean(), covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.no_grad():
                fit_gpytorch_mll(mll)

            acq_func = qNoisyExpectedImprovement(model, train_X)
            
            candidates, _ = optimize_acqf(
                acq_func, 
                bounds=torch.stack([lb, ub]), 
                q=1, 
                num_restarts=10, 
                raw_samples=512
            )
            
            new_x = candidates.detach().squeeze(0).numpy()
            new_y = func(new_x)
            
            X_list.append(torch.tensor(new_x, dtype=torch.double))
            Y_list.append(torch.tensor([new_y], dtype=torch.double))
            
            if new_y < best_f:
                best_f = new_y
                best_x = new_x

        return best_f, best_x