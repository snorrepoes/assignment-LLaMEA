import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean

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
        
        X_obs = torch.tensor(self.initial_points, dtype=torch.double)
        y_obs = torch.tensor([func(x) for x in X_obs], dtype=torch.double).unsqueeze(-1)
        
        if X_obs.shape[0] > self.budget:
            raise ValueError("Initial points exceed budget.")

        best_f = y_obs.min()
        best_x = X_obs[y_obs.argmin()]

        for i in range(len(X_obs), self.budget):
            train_Y = (y_obs - y_obs.mean()) / y_obs.std()
            
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X_obs, train_Y, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            qNEI = qNoisyExpectedImprovement(model, X_observation=X_obs)
            
            new_x, _ = optimize_acqf(
                qNEI, bounds=torch.stack([lb, ub]), q=1, num_restarts=10, raw_samples=512
            )
            
            new_f = func(new_x.squeeze().numpy())
            
            X_obs = torch.cat([X_obs, new_x])
            y_obs = torch.cat([y_obs, torch.tensor([[new_f]], dtype=torch.double)])
            
            if new_f < best_f:
                best_f = new_f
                best_x = new_x.squeeze()

        return best_f.item(), best_x.numpy()