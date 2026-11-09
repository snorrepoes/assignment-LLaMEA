import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogNoisyExpectedImprovement
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
        
        length_min, length_max = 0.1, 4.0
        length = 1.0
        best_x = train_x[np.argmin(train_y)]
        best_val = min(train_y)
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Log-transform targets: -log(y - min_y + eps) is numerically unstable if not careful, 
            # use a simpler approach: Standardize and fit GP to log-space
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            Y_log = torch.log(Y - Y.min() + 1e-9)
            
            model = SingleTaskGP(X, Y_log, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Trust Region boundaries
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            # Use qLogNEI for better plateau handling
            acq = qLogNoisyExpectedImprovement(model, X_baseline=X, prune_baseline=True)
            new_x, _ = optimize_acqf(acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, 
                                     num_restarts=10, raw_samples=512)
            
            new_x = new_x.detach().flatten()
            new_f = float(func(new_x.numpy()))
            
            train_x.append(new_x)
            train_y.append(new_f)
            
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                length = min(length * 1.2, length_max)
            else:
                length = max(length * 0.5, length_min)

        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()