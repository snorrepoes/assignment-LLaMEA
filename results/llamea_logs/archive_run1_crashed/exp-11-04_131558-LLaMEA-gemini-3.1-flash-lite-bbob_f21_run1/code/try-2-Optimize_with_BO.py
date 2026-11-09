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
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        calls = len(train_x)
        if calls > self.budget:
            raise ValueError("Initial points exceed budget.")

        while calls < self.budget:
            # Model definition with Matern 5/2 kernel for smoothness flexibility
            model = SingleTaskGP(train_x, train_y, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Use qNoisyExpectedImprovement for robust exploration in multimodal spaces
            acq_func = qNoisyExpectedImprovement(model, X_baseline=train_x)
            
            # Optimize acquisition function
            new_x, _ = optimize_acqf(
                acq_func, bounds=torch.stack([lb, ub]), q=1, num_restarts=10, raw_samples=512
            )
            
            new_y = torch.tensor([func(new_x.squeeze().numpy())], dtype=torch.double).unsqueeze(-1)
            
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])
            calls += 1

        best_idx = torch.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()