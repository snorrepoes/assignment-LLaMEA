import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

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
        bounds = torch.stack([lb, ub])
        
        X_obs = torch.tensor(self.initial_points, dtype=torch.double)
        y_obs = torch.tensor([func(x) for x in X_obs.numpy()], dtype=torch.double).unsqueeze(-1)
        
        if X_obs.shape[0] > self.budget:
            raise ValueError("Initial points exceed budget.")

        for _ in range(len(X_obs), self.budget):
            # Normalize outcomes for better GP convergence
            y_mean, y_std = y_obs.mean(), y_obs.std()
            y_std = y_std if y_std > 0 else 1.0
            train_Y = (y_obs - y_mean) / y_std
            
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X_obs, train_Y, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.no_grad():
                fit_gpytorch_mll(mll)

            qNEI = qNoisyExpectedImprovement(model, X_baseline=X_obs)
            
            new_x, _ = optimize_acqf(
                qNEI, bounds=bounds, q=1, num_restarts=10, raw_samples=512
            )
            
            new_f = func(new_x.detach().squeeze().numpy())
            
            X_obs = torch.cat([X_obs, new_x.detach()])
            y_obs = torch.cat([y_obs, torch.tensor([[new_f]], dtype=torch.double)])

        best_idx = y_obs.argmin()
        return y_obs[best_idx].item(), X_obs[best_idx].numpy()