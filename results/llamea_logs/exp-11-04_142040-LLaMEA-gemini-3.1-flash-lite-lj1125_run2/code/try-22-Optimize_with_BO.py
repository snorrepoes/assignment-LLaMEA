import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize

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
        
        # TuRBO-1 parameters
        length_min = 0.5
        length_max = 2.0
        length = 1.6
        success_counter = 0
        failure_counter = 0
        success_tol = 3
        failure_tol = 5

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Log transform Y to handle high dynamic range of objective
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            Y_log = torch.log(Y - Y.min() + 1e-9)
            
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Trust region bounds
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            model = SingleTaskGP(
                X, Y_log, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq = qLogNoisyExpectedImprovement(model, X_baseline=X)
            new_x, _ = optimize_acqf(acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, num_restarts=10, raw_samples=512)
            
            new_f = func(new_x.detach().numpy()[0])
            
            # Update TuRBO state
            if new_f < Y.min().item():
                success_counter += 1
                failure_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
            
            if success_counter >= success_tol:
                length = min(length_max, length * 2.0)
                success_counter = 0
            elif failure_counter >= failure_tol:
                length /= 2.0
                failure_counter = 0
            
            if length < length_min:
                length = length_max # Reset trust region if too small
            
            train_x.append(new_x.squeeze())
            train_y.append(new_f)

        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()