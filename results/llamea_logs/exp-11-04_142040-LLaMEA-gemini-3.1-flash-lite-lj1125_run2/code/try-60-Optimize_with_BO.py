import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.generation import MaxPosteriorSampling
from botorch.models.gp_regression import GP

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
        length_min, length_max = 0.05, 4.0
        success_counter, failure_counter = 0, 0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            
            # Log transform Y to handle high dynamic range and flat plateaus
            Y_trans = torch.log(Y - Y.min() + 1e-9)
            
            model = SingleTaskGP(X, Y_trans, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Thompson Sampling inside a trust region with fallback to global uniform
            if len(train_x) % 15 == 0:
                tr_lb, tr_ub = lb, ub
            else:
                tr_lb = torch.clamp(best_x - length, lb, ub)
                tr_ub = torch.clamp(best_x + length, lb, ub)
            
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            candidate = sampler(torch.rand(2048, self.dim) * (tr_ub - tr_lb) + tr_lb, num_samples=1)
            new_x = candidate.detach().flatten()
            
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
                
            if success_counter >= 3:
                length = min(length * 1.5, length_max)
                success_counter = 0
            elif failure_counter >= 4:
                length = max(length / 2.0, length_min)
                failure_counter = 0
        
        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()