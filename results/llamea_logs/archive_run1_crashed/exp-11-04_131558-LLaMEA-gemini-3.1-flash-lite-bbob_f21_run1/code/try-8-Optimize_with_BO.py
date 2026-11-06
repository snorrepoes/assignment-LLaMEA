import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qLogNoisyExpectedImprovement
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
        self.bounds = torch.tensor([[-5.0] * dim, [5.0] * dim], dtype=torch.double)

    def __call__(self, func):
        X_obs = torch.tensor(self.initial_points, dtype=torch.double)
        y_obs = torch.tensor([func(x) for x in X_obs], dtype=torch.double).unsqueeze(-1)
        
        budget_used = len(X_obs)
        if budget_used > self.budget:
            raise Exception("Initial points exceed budget.")

        while budget_used < self.budget:
            # Fit GP with Matern 2.5 kernel
            model = SingleTaskGP(X_obs, y_obs)
            model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Use qLogNoisyExpectedImprovement for better landscape exploration
            qLogNEI = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_obs,
                prune_baseline=True
            )

            # Optimize acquisition function with restarts
            new_x, _ = optimize_acqf(
                acq_function=qLogNEI,
                bounds=self.bounds,
                q=1,
                num_restarts=15,
                raw_samples=512,
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X_obs = torch.cat([X_obs, new_x])
            y_obs = torch.cat([y_obs, torch.tensor([[new_y]], dtype=torch.double)])
            budget_used += 1

        best_idx = torch.argmin(y_obs).item()
        return y_obs[best_idx].item(), X_obs[best_idx].numpy()