import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import LogStandardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

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
        
        if len(X) > self.budget:
            raise ValueError("Budget exceeded")

        while len(X) < self.budget:
            train_Y = Y
            
            model = SingleTaskGP(
                X, train_Y, 
                outcome_transform=LogStandardize(),
                input_transform=Normalize(d=self.dim, bounds=bounds)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Use qNoisyExpectedImprovement to robustly handle the objective landscape
            acq_func = qNoisyExpectedImprovement(
                model=model,
                X_baseline=X,
                prune_baseline=True
            )
            
            new_x, _ = optimize_acqf(
                acq_func,
                bounds=bounds,
                q=1,
                num_restarts=15,
                raw_samples=1024,
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, torch.tensor([[new_y]], dtype=torch.double)])
            
        best_idx = torch.argmin(Y)
        return Y[best_idx].item(), X[best_idx].numpy()