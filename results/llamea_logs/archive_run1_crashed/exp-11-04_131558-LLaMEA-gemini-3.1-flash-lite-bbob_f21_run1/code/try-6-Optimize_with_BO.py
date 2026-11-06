import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
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
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        curr_budget = len(train_x)
        if curr_budget > self.budget:
            raise ValueError("Initial points exceed budget.")

        while curr_budget < self.budget:
            # Standardize y to improve numerical stability for the GP
            train_y_std = (train_y - train_y.mean()) / (train_y.std() + 1e-8)
            
            # Use a robust kernel with Gamma priors to stabilize hyperparameter estimation
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim, 
                                             lengthscale_prior=GammaPrior(3.0, 6.0)))
            likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))
            
            model = SingleTaskGP(train_x, train_y_std, covar_module=kernel, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.no_grad():
                fit_gpytorch_mll(mll)

            # qNoisyExpectedImprovement is excellent for multimodal landscapes
            acq_func = qNoisyExpectedImprovement(model, X_baseline=train_x)
            
            # Use more raw samples and restarts to navigate the complex multimodal landscape
            new_x, _ = optimize_acqf(
                acq_func, bounds=torch.stack([lb, ub]), q=1, num_restarts=15, raw_samples=1024
            )
            
            new_y = torch.tensor([func(new_x.detach().squeeze().numpy())], dtype=torch.double).unsqueeze(-1)
            
            train_x = torch.cat([train_x, new_x.detach()])
            train_y = torch.cat([train_y, new_y])
            curr_budget += 1

        best_idx = torch.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()