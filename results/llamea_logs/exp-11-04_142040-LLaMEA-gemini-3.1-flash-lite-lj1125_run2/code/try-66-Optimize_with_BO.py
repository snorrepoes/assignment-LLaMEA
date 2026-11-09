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
        
        length = 2.0
        min_length = 0.01
        max_length = 4.0
        success_streak = 0
        failure_streak = 0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y_raw = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            # Log transform Y to handle massive dynamic range and flat plateaus
            Y = torch.log(Y_raw - Y_raw.min() + 1e-12)
            
            model = SingleTaskGP(
                X, Y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = torch.argmin(Y_raw)
            best_x = X[best_idx]
            
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qLogNoisyExpectedImprovement(model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = func(new_x_val.numpy())
            
            if new_f < train_y[best_idx]:
                success_streak += 1
                failure_streak = 0
                if success_streak >= 3:
                    length = min(length * 2.0, max_length)
                    success_streak = 0
            else:
                success_streak = 0
                failure_streak += 1
                if failure_streak >= 2:
                    length = max(length * 0.5, min_length)
                    failure_streak = 0
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()