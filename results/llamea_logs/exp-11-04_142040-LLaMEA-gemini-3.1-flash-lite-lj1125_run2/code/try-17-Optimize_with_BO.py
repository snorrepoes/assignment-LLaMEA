import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb, ub = -5.0, 5.0
        bounds = torch.tensor([[lb] * self.dim, [ub] * self.dim], dtype=torch.double)
        
        X = torch.tensor(self.initial_points, dtype=torch.double)
        Y = torch.tensor([func(x) for x in X.numpy()], dtype=torch.double).view(-1, 1)
        
        # TuRBO state variables
        length = 1.6
        best_idx = torch.argmin(Y)
        best_val = Y[best_idx]
        best_x = X[best_idx]
        
        curr_budget = len(X)
        while curr_budget < self.budget:
            # Constrain search to Trust Region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            # Model with outcome normalization
            model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            acq = qLogNoisyExpectedImprovement(model, X_baseline=X, sampler=None)
            
            new_x, _ = optimize_acqf(
                acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, num_restarts=10, raw_samples=128
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, torch.tensor([[new_y]], dtype=torch.double)])
            curr_budget += 1
            
            # Update TuRBO local search
            if new_y < best_val:
                best_val = new_y
                best_x = new_x
                length = min(10.0, length * 1.1)
            else:
                length = length * 0.95
                if length < 1e-3: length = 1.6 # Reset if trapped
        
        final_best_idx = torch.argmin(Y)
        return Y[final_best_idx].item(), X[final_best_idx].numpy()