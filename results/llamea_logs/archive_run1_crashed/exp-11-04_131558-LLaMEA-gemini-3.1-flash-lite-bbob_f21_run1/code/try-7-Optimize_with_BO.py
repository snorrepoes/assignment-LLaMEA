import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.ucb import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan

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
        X_obs = torch.tensor(self.initial_points, dtype=torch.double)
        y_obs = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        budget_left = self.budget - len(self.initial_points)
        if budget_left < 0:
            raise ValueError("Initial points exceed budget")

        bounds = torch.tensor([[self.lb]*self.dim, [self.ub]*self.dim], dtype=torch.double)
        
        for i in range(budget_left):
            # Use Matern 2.5 kernel for smooth approximation, enable ARD
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X_obs, y_obs, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.no_grad():
                fit_gpytorch_mll(mll)

            # Beta decay for UCB exploration vs exploitation balance
            beta = 2.0 * (1.0 - (i / self.budget))
            acq_func = qUpperConfidenceBound(model, beta=beta)

            new_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=15,
                raw_samples=512,
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X_obs = torch.cat([X_obs, new_x])
            y_obs = torch.cat([y_obs, torch.tensor([[new_y]], dtype=torch.double)])

        best_idx = torch.argmin(y_obs).item()
        return y_obs[best_idx].item(), X_obs[best_idx].numpy()