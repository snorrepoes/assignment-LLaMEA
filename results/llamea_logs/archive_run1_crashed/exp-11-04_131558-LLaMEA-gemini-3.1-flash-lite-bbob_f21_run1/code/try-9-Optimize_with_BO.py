import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.models import ExactGP
from botorch.models.transforms import Normalize, Standardize

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
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        calls = len(train_x)
        if calls > self.budget:
            raise ValueError("Initial points exceed budget.")

        while calls < self.budget:
            # Using Normalize and Standardize transforms to handle numerical instability and scaling
            model = SingleTaskGP(
                train_x, 
                train_y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.warnings.catch_warnings():
                torch.warnings.simplefilter("ignore")
                fit_gpytorch_mll(mll)

            # qLogNoisyExpectedImprovement is generally more numerically stable than qNEI
            acq_func = qLogNoisyExpectedImprovement(
                model=model, 
                X_baseline=train_x,
                prune_baseline=True
            )
            
            # Use a conservative number of restarts to avoid gradient issues
            new_x, _ = optimize_acqf(
                acq_func, 
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]), 
                q=1, 
                num_restarts=8, 
                raw_samples=256
            )
            
            # Map normalized optimization domain back to [lb, ub]
            actual_new_x = lb + (ub - lb) * new_x.squeeze()
            new_y = torch.tensor([func(actual_new_x.numpy())], dtype=torch.double).unsqueeze(-1)
            
            train_x = torch.cat([train_x, actual_new_x.unsqueeze(0)])
            train_y = torch.cat([train_y, new_y])
            calls += 1

        best_idx = torch.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()