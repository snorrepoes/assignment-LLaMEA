import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch.quasirandom import SobolEngine

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
        
        length = 2.0
        min_length = 0.5
        max_length = 4.0
        
        while len(train_x) < self.budget:
            # Fit model
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Local trust region around best
            best_idx = torch.argmin(train_y)
            best_x = train_x[best_idx]
            
            tr_lb = torch.max(lb, best_x - length)
            tr_ub = torch.min(ub, best_x + length)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            # Use qNoisyExpectedImprovement for robust selection
            acq = qNoisyExpectedImprovement(model, X_baseline=train_x)
            
            candidate, _ = optimize_acqf(
                acq, bounds=tr_bounds, q=1, num_restarts=10, raw_samples=512
            )
            
            new_x = candidate.detach().flatten()
            new_y = func(new_x.numpy())
            
            # Logic to update TR
            if new_y < train_y[best_idx]:
                length = min(length * 1.1, max_length)
            else:
                length = max(length * 0.8, min_length)
                # Restart TR if we keep failing
                if length <= min_length:
                    length = 2.0
            
            train_x = torch.cat([train_x, new_x.unsqueeze(0)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()