import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
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
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).view(-1, 1)
        
        num_calls = len(train_x)
        if num_calls > self.budget:
            raise ValueError("Initial points exceed budget.")

        # TuRBO-1 state variables
        length = 0.8
        min_length = 0.01
        max_length = 1.6
        success_tol = 3
        failure_tol = 5
        success_count = 0
        failure_count = 0
        
        while num_calls < self.budget:
            train_x_norm = (train_x - lb) / (ub - lb)
            
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1),
                covar_module=covar_module
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_val = train_y.min()
            best_x = train_x[train_y.argmin()]
            
            # Trust region bounds
            tr_lb = torch.clamp(best_x - length * (ub - lb) / 2, lb, ub)
            tr_ub = torch.clamp(best_x + length * (ub - lb) / 2, lb, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            log_ei = LogExpectedImprovement(model, best_f=best_val)
            candidate, _ = optimize_acqf(
                acq_function=log_ei,
                bounds=tr_bounds,
                q=1,
                num_restarts=10,
                raw_samples=256,
            )

            new_x = candidate.detach()
            new_y = func(new_x.flatten().numpy())
            
            if new_y < best_val:
                success_count += 1
                failure_count = 0
            else:
                success_count = 0
                failure_count += 1
                
            if success_count >= success_tol:
                length = min(2.0 * length, max_length)
                success_count = 0
            elif failure_count >= failure_tol:
                length /= 2.0
                failure_count = 0
                
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1
            
            if length < min_length:
                break

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()