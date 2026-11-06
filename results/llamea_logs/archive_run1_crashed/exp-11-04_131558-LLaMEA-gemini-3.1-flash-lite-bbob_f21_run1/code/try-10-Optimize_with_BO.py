import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
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
        
        train_X = torch.tensor(self.initial_points, dtype=torch.double)
        train_Y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        while len(train_X) < self.budget:
            # We minimize, so we negate Y for Expected Improvement which expects maximization
            model = SingleTaskGP(train_X, -train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            best_f = (-train_Y).max()
            acq_func = LogExpectedImprovement(model=model, best_f=best_f)
            
            new_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            new_x_np = new_x.detach().numpy().flatten()
            new_y = func(new_x_np)
            
            train_X = torch.cat([train_X, new_x.detach()])
            train_Y = torch.cat([train_Y, torch.tensor([[new_y]], dtype=torch.double)])
            
        best_idx = torch.argmin(train_Y)
        return float(train_Y[best_idx]), train_X[best_idx].numpy()