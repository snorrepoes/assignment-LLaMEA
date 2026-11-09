import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import LogStandardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Normalize

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
        
        train_x = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        train_y = [func(x) for x in self.initial_points]
        
        if len(train_x) > self.budget:
            raise ValueError("Initial points exceed budget.")

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Using LogStandardize transform to handle dynamic range and plateaus
            model = SingleTaskGP(
                X, Y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=LogStandardize()
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # qNEI is effective for noisy black-box optimization
            acqf = qNoisyExpectedImprovement(model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                acqf, 
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]), 
                q=1, 
                num_restarts=20, 
                raw_samples=512
            )
            
            # Denormalize
            new_x = new_x.squeeze(0)
            actual_x = new_x * (ub - lb) + lb
            
            new_f = func(actual_x.detach().numpy())
            
            train_x.append(actual_x)
            train_y.append(new_f)
            
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()