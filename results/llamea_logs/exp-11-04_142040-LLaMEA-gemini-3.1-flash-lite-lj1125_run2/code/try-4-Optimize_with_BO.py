import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.models.transforms import Normalize, Standardize

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([[-5.0] * self.dim, [5.0] * self.dim], dtype=torch.double)
        X = torch.tensor(self.initial_points, dtype=torch.double)
        Y = torch.tensor([func(x) for x in X], dtype=torch.double).unsqueeze(-1)
        
        count = len(X)
        if count > self.budget:
            raise ValueError("Budget exceeded")

        while count < self.budget:
            # Use a log transformation to handle the flat plateau vs narrow well dynamic range
            # We add a small offset to ensure positive values
            y_min = Y.min()
            train_Y = torch.log(Y - y_min + 1e-12)
            
            # GP model with input normalization and output standardization
            model = SingleTaskGP(X, train_Y, input_transform=Normalize(self.dim), 
                                 outcome_transform=Standardize(1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Use qNoisyExpectedImprovement which is robust for black-box tasks
            qNEI = qNoisyExpectedImprovement(model, X_baseline=X)
            
            # Optimize acquisition
            new_x, _ = optimize_acqf(
                qNEI, bounds=bounds, q=1, num_restarts=20, raw_samples=1024
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, torch.tensor([[new_y]], dtype=torch.double)])
            count += 1
            
        best_idx = torch.argmin(Y)
        return Y[best_idx].item(), X[best_idx].numpy()