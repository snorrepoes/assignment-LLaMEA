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
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        budget_left = self.budget
        X_data = torch.tensor(self.initial_points, dtype=torch.double)
        Y_data = []

        for x in X_data:
            if budget_left <= 0: break
            Y_data.append(func(x.numpy()))
            budget_left -= 1
        
        Y_data = torch.tensor(Y_data, dtype=torch.double).unsqueeze(-1)
        
        # Track current best
        best_f = Y_data.min().item()
        best_x = X_data[Y_data.argmin()].numpy()

        while budget_left > 0:
            # Transform Y: Log transform to handle flat plateaus/dynamic range
            Y_transformed = torch.log(Y_data - Y_data.min() + 1e-9)
            
            # Setup Model
            likelihood = GaussianLikelihood()
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X_data, Y_transformed, likelihood=likelihood, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Acquisition Function
            acq_func = qNoisyExpectedImprovement(model=model, X_baseline=X_data)
            
            # Optimize Acquisition
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[self.lb]*self.dim, [self.ub]*self.dim], dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=128
            )
            
            new_x = candidates.detach()
            new_y = func(new_x.squeeze().numpy())
            budget_left -= 1
            
            X_data = torch.cat([X_data, new_x])
            Y_data = torch.cat([Y_data, torch.tensor([[new_y]], dtype=torch.double)])
            
            if new_y < best_f:
                best_f = new_y
                best_x = new_x.squeeze().numpy()
                
        return best_f, best_x