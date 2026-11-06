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
        X_train = torch.tensor(self.initial_points, dtype=torch.double)
        Y_train = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        curr_budget = len(self.initial_points)
        
        while curr_budget < self.budget:
            # Add jitter to Y if values are identical to avoid Cholesky issues
            y_std = Y_train.std()
            y_train_norm = (Y_train - Y_train.mean()) / (y_std + 1e-9)
            
            likelihood = GaussianLikelihood(noise_constraint=torch.distributions.constraints.GreaterThan(1e-4))
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X_train, y_train_norm, likelihood=likelihood, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            try:
                fit_gpytorch_mll(mll)
            except:
                # Fallback to standard initialization if fit fails
                pass

            qNEI = qNoisyExpectedImprovement(model=model, X_baseline=X_train)
            
            new_x, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X_train = torch.cat([X_train, new_x])
            Y_train = torch.cat([Y_train, torch.tensor([[new_y]], dtype=torch.double)])
            curr_budget += 1

        best_idx = torch.argmin(Y_train)
        return float(Y_train[best_idx]), X_train[best_idx].numpy()