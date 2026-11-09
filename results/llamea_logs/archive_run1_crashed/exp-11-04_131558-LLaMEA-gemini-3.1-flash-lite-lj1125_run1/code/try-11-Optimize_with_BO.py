import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

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
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        if len(train_x) > self.budget:
            raise ValueError("Initial points exceed budget.")

        while len(train_x) < self.budget:
            # Standardize and handle potential log-scale features via outcome transform
            # We use Log transform to flatten the deep well landscape for the GP
            train_y_log = torch.log(train_y - train_y.min() + 1e-12)
            
            model = SingleTaskGP(
                train_x, 
                train_y_log, 
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            qNEI = qNoisyExpectedImprovement(model, X_baseline=train_x)
            
            # Use high number of restarts to find the narrow global minimum
            new_x, _ = optimize_acqf(
                qNEI, 
                bounds=bounds, 
                q=1, 
                num_restarts=20, 
                raw_samples=512
            )
            
            new_f = func(new_x.detach().numpy()[0])
            
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, torch.tensor([[new_f]], dtype=torch.double)])
            
        best_idx = torch.argmin(train_y).item()
        return float(train_y[best_idx]), train_x[best_idx].numpy()