import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.transforms import unnormalize, normalize

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
        
        train_x = []
        train_y = []
        
        for x in self.initial_points:
            train_x.append(torch.tensor(x, dtype=torch.double))
            train_y.append(float(func(x)))
            
        # TuRBO-1 parameters
        length = 0.8 * (ub - lb)
        best_x = train_x[np.argmin(train_y)]
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Log transform Y to handle high dynamic range and flat plateaus
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            Y_log = torch.log(Y - Y.min() + 1e-9)
            
            model = SingleTaskGP(
                X, Y_log, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Local search within trust region
            tr_bounds = torch.stack([
                torch.max(lb, best_x - length/2),
                torch.min(ub, best_x + length/2)
            ])
            
            acqf = qLogNoisyExpectedImprovement(model, X_baseline=X)
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=tr_bounds,
                q=1,
                num_restarts=10,
                raw_samples=256,
            )
            
            new_x = new_x.detach().flatten()
            new_f = func(new_x.numpy())
            
            train_x.append(new_x)
            train_y.append(new_f)
            
            if new_f < min(train_y[:-1]):
                best_x = new_x
                length = min(length * 1.1, 2.0 * (ub - lb))
            else:
                length = max(length * 0.8, 1e-3 * (ub - lb))

        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()