import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.generation import MaxPosteriorSampling

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
        train_y = [float(func(x)) for x in self.initial_points]
        
        best_val = min(train_y)
        best_x = train_x[np.argmin(train_y)]
        
        length = 2.0
        length_min, length_max = 0.1, 4.0
        success_counter, failure_counter = 0, 0
        success_tol, failure_tol = 3, 5

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            # Log transform to handle flat plateaus
            Y_trans = -torch.log(Y - Y.min() + 1e-9)
            
            model = SingleTaskGP(X, Y_trans, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Thompson Sampling within Trust Region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            new_x = sampler(torch.rand(1024, self.dim, device=lb.device) * (tr_ub - tr_lb) + tr_lb, num_samples=1)
            new_x = new_x.detach().flatten()
            
            new_f = float(func(new_x.numpy()))
            train_x.append(new_x)
            train_y.append(new_f)
            
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                success_counter += 1
                failure_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
                
            if success_counter >= success_tol:
                length = min(length * 2.0, length_max)
                success_counter = 0
            elif failure_counter >= failure_tol:
                length = max(length / 2.0, length_min)
                failure_counter = 0
        
        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()