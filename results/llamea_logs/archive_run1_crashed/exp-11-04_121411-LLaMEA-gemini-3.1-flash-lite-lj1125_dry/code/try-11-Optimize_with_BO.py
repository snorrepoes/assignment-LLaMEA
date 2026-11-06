import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

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
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        if len(train_x) > self.budget:
            raise ValueError("Initial points exceed budget.")

        while len(train_x) < self.budget:
            # Use Standardize for target and Normalize for inputs to improve GP stability
            model = SingleTaskGP(
                train_x, 
                train_y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            try:
                fit_gpytorch_mll(mll)
            except:
                # Fallback to simple mean if fitting fails
                pass
            
            acq_func = qNoisyExpectedImprovement(
                model=model, 
                X_baseline=train_x,
                prune_baseline=True
            )
            
            new_x, _ = optimize_acqf(
                acq_func, 
                bounds=torch.stack([lb, ub]), 
                q=1, 
                num_restarts=10, 
                raw_samples=256
            )
            
            new_x_np = new_x.detach().numpy()[0]
            new_f = func(new_x_np)
            
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, torch.tensor([[new_f]], dtype=torch.double)])
            
        best_idx = torch.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()