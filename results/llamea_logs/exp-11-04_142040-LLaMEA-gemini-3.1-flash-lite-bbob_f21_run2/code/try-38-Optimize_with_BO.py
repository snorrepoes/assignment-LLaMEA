import torch
import numpy as np
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

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
        
        length = 2.5
        while len(train_x) < self.budget:
            # Normalize inputs for better GP performance
            train_x_norm = (train_x - lb) / (ub - lb)
            
            # Using a simple GP with Matern 5/2 kernel
            model = SingleTaskGP(
                train_x, train_y, 
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            )
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define TR around current best
            best_idx = torch.argmin(train_y)
            best_x = train_x[best_idx]
            
            tr_lb = torch.max(lb, best_x - length)
            tr_ub = torch.min(ub, best_x + length)
            
            # Use qLogExpectedImprovement for better stability in sharp basins
            acq = qLogExpectedImprovement(model, best_f=train_y.min())
            
            candidate, _ = optimize_acqf(
                acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, num_restarts=15, raw_samples=1024
            )
            
            new_x = candidate.detach().flatten()
            new_y = func(new_x.numpy())
            
            # Update TR: Dynamic shrinking/expansion
            if new_y < train_y[best_idx]:
                length = min(length * 1.05, 5.0)
            else:
                length = max(length * 0.9, 0.2)
            
            train_x = torch.cat([train_x, new_x.unsqueeze(0)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()