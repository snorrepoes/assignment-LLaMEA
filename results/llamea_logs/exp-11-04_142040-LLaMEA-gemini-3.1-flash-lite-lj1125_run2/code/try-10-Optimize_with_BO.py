import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
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

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Model with standardization and Matern kernel
            model = SingleTaskGP(
                X, Y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Use qLogNoisyExpectedImprovement for better handling of deep narrow wells
            qLogNEI = qLogNoisyExpectedImprovement(model, X_baseline=X)
            
            # Constrain search area around current best for local refinement
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            tr_bounds = torch.stack([
                torch.max(lb, best_x - 0.5),
                torch.min(ub, best_x + 0.5)
            ])
            
            new_x, _ = optimize_acqf(
                acq_function=qLogNEI,
                bounds=tr_bounds,
                q=1,
                num_restarts=15,
                raw_samples=512,
            )
            
            new_x = new_x.detach().numpy()[0]
            new_f = func(new_x)
            
            train_x.append(torch.tensor(new_x, dtype=torch.double))
            train_y.append(new_f)

        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()