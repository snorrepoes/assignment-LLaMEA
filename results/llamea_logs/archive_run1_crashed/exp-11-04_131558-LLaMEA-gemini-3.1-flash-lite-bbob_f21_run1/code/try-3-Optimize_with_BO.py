import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        budget_left = self.budget
        X_obs = []
        y_obs = []

        # Evaluate initial points
        for x in self.initial_points:
            if budget_left <= 0: break
            X_obs.append(x)
            y_obs.append(func(x))
            budget_left -= 1
        
        train_X = torch.tensor(np.array(X_obs), dtype=torch.double)
        train_Y = torch.tensor(np.array(y_obs), dtype=torch.double).unsqueeze(-1)

        while budget_left > 0:
            # Setup Model: Matern kernel is robust for multimodal landscapes
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Acquisition: qNoisyExpectedImprovement handles arbitrary test points
            qNEI = qNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                prune_baseline=True
            )

            # Optimization of acquisition function
            new_x, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.tensor([[self.lb]*self.dim, [self.ub]*self.dim], dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=200
            )
            
            new_x_np = new_x.detach().numpy().flatten()
            new_y = func(new_x_np)
            
            train_X = torch.cat([train_X, new_x])
            train_Y = torch.cat([train_Y, torch.tensor([[new_y]], dtype=torch.double)])
            budget_left -= 1

        best_idx = torch.argmin(train_Y).item()
        return train_Y[best_idx].item(), train_X[best_idx].numpy()