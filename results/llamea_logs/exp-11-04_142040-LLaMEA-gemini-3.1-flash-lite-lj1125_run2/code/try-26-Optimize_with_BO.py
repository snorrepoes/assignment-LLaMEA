import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
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
        train_y = [float(func(x)) for x in self.initial_points]

        best_x = train_x[np.argmin(train_y)]
        best_val = min(train_y)
        
        # Length scale parameters
        length = 1.0
        length_min, length_max = 0.05, 2.0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Log transform Y to handle high dynamic range and flat plateaus
            # We use Y_log = -log(Y - min_Y + eps) to convert minimization to maximization
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            y_min = Y.min()
            Y_log = -torch.log(Y - y_min + 1e-9)
            
            model = SingleTaskGP(X, Y_log, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define localized trust region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            # Use Log Expected Improvement for high dynamic range
            acq = qLogExpectedImprovement(model, best_f=Y_log.max())
            
            new_x, _ = optimize_acqf(
                acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, 
                num_restarts=15, raw_samples=512
            )
            
            new_x = new_x.detach().flatten()
            new_f = float(func(new_x.numpy()))
            
            train_x.append(new_x)
            train_y.append(new_f)
            
            # Update best and adaptive trust region
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                length = min(length * 1.2, length_max)
            else:
                length = max(length * 0.7, length_min)
                
        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()