import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.lb = -5.0
        self.ub = 5.0
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim], dtype=torch.double)
        
        X = torch.tensor(self.initial_points, dtype=torch.double)
        Y = torch.tensor([func(x) for x in X], dtype=torch.double).unsqueeze(-1)
        
        count = len(X)
        if count > self.budget:
            raise ValueError("Initial points exceed budget")

        while count < self.budget:
            # Shift Y to be positive and apply log to emphasize the narrow well structure
            y_min = Y.min()
            train_Y = torch.log(Y - y_min + 1e-9)
            
            # Normalize inputs to [0, 1] for stable GP fitting
            train_X = normalize(X, bounds)
            
            # Model with Matern Kernel
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Acquisition: qNoisyExpectedImprovement
            qNEI = qNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                prune_baseline=True
            )
            
            # Constrained optimization to search in normalized space
            unit_bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
            new_x_norm, _ = optimize_acqf(
                qNEI, bounds=unit_bounds, q=1, num_restarts=15, raw_samples=1024
            )
            
            new_x = unnormalize(new_x_norm, bounds)
            new_y = func(new_x.detach().numpy().flatten())
            
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, torch.tensor([[new_y]], dtype=torch.double)])
            count += 1
            
        best_idx = torch.argmin(Y)
        return Y[best_idx].item(), X[best_idx].numpy()