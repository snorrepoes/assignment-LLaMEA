import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms.outcome import Standardize

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
        
        train_x = []
        train_y = []
        
        # Evaluate initial points
        for x in self.initial_points:
            if len(train_x) >= self.budget:
                break
            y = func(x)
            train_x.append(torch.tensor(x, dtype=torch.double))
            train_y.append(float(y))
            
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Apply log transform to objective values to handle dynamic range and plateaus
            Y = torch.log(torch.tensor(train_y, dtype=torch.double).unsqueeze(-1) - torch.min(torch.tensor(train_y)) + 1e-9)
            
            # Use Matern 5/2 kernel for smoother modeling of the landscape
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                X, Y, 
                likelihood=GaussianLikelihood(),
                covar_module=covar_module,
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            qNEI = qNoisyExpectedImprovement(model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            x_next = new_x.detach().squeeze(0).numpy()
            y_next = func(x_next)
            
            train_x.append(new_x.squeeze(0))
            train_y.append(float(y_next))
            
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()