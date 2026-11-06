import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.noisy_expected_improvement import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

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
        
        n_evals = len(Y)
        if n_evals > self.budget:
            raise ValueError("Initial points exceed budget")

        while n_evals < self.budget:
            # Use log-transform for objective to handle deep wells and standardization for better GP convergence
            # Using log1p(y - min_y) to ensure values are positive and capture relative magnitude differences
            min_y = Y.min()
            train_Y = torch.log1p(Y - min_y)
            
            model = SingleTaskGP(
                train_X=X, 
                train_Y=train_Y, 
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=self.dim, bounds=bounds)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Use qNoisyExpectedImprovement as it is robust to observation noise and flat landscapes
            acq_func = qNoisyExpectedImprovement(model=model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, torch.tensor([[new_y]], dtype=torch.double)])
            n_evals += 1
            
        best_idx = torch.argmin(Y)
        return Y[best_idx].item(), X[best_idx].numpy()