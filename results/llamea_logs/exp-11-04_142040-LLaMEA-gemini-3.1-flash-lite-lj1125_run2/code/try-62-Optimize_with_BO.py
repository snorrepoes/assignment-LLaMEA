import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
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
        
        best_idx = np.argmin(train_y)
        best_x = train_x[best_idx]
        best_val = train_y[best_idx]
        
        length = 0.5 * (ub - lb)
        length_min, length_max = 0.01, 2.0
        success_streak = 0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            # Log-transform to handle high dynamic range and flat plateaus
            Y_log = torch.log(Y - Y.min() + 1e-9)
            
            model = SingleTaskGP(X, Y_log, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Use Thompson Sampling for better exploration within the trust region
            sampler = MaxPosteriorSampling(model, replacement=False)
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            new_x = sampler(torch.stack([tr_lb, tr_ub]), num_samples=100)
            new_x = new_x.detach().flatten(0, 1)[0]
            
            new_f = float(func(new_x.numpy()))
            train_x.append(new_x)
            train_y.append(new_f)
            
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                success_streak += 1
                if success_streak >= 3:
                    length = torch.clamp(length * 1.5, max=length_max)
                    success_streak = 0
            else:
                length = torch.clamp(length * 0.7, min=length_min)
                success_streak = 0
                
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()