import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

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
        bounds = torch.stack([lb, ub])
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        # Hyperparameters for multi-region strategy
        length = 2.0
        success_count = 0
        failure_count = 0
        
        while len(train_x) < self.budget:
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = torch.argmin(train_y)
            best_val = train_y[best_idx]
            best_x = train_x[best_idx]
            
            # Dynamic Trust Region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            acq = qExpectedImprovement(model, best_f=best_val)
            
            candidate, _ = optimize_acqf(
                acq, bounds=tr_bounds, q=1, num_restarts=15, raw_samples=1024
            )
            
            new_x = candidate.detach().flatten()
            new_y = torch.tensor([[func(new_x.numpy())]], dtype=torch.double)
            
            train_x = torch.cat([train_x, new_x.unsqueeze(0)])
            train_y = torch.cat([train_y, new_y])
            
            # Update TR length based on progress
            if new_y < best_val:
                success_count += 1
                failure_count = 0
                if success_count >= 3:
                    length = min(length * 1.2, 4.0)
                    success_count = 0
            else:
                failure_count += 1
                success_count = 0
                if failure_count >= 5:
                    length = max(length * 0.5, 0.2)
                    failure_count = 0
                    
        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()