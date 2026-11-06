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
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.double)
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        calls = len(train_x)
        if calls > self.budget:
            raise ValueError("Initial points exceed budget.")

        while calls < self.budget:
            # Use a robust GP with a noise-aware likelihood
            likelihood = GaussianLikelihood(noise_constraint=torch.tensor(1e-4))
            model = SingleTaskGP(train_x, train_y, likelihood=likelihood, 
                                 covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.no_grad():
                fit_gpytorch_mll(mll)

            acq_func = qNoisyExpectedImprovement(model, X_baseline=train_x, prune_baseline=True)
            
            # Use a higher raw_samples count and constraints to improve numerical stability
            new_x, _ = optimize_acqf(
                acq_func, 
                bounds=bounds, 
                q=1, 
                num_restarts=15, 
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200}
            )
            
            new_y = torch.tensor([func(new_x.detach().squeeze().numpy())], dtype=torch.double).unsqueeze(-1)
            
            train_x = torch.cat([train_x, new_x.detach()])
            train_y = torch.cat([train_y, new_y])
            calls += 1

        best_idx = torch.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()