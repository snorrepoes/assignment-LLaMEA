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
        
        # Initialize containers
        X_train = torch.tensor(self.initial_points, dtype=torch.double)
        Y_train = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        curr_budget = len(self.initial_points)
        if curr_budget > self.budget:
            raise ValueError("Initial points exceed budget.")

        while curr_budget < self.budget:
            # Fit GP
            likelihood = GaussianLikelihood()
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X_train, Y_train, likelihood=likelihood, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            fit_gpytorch_mll(mll)

            # Define Acquisition
            qNEI = qNoisyExpectedImprovement(model=model, X_baseline=X_train)
            
            # Optimize Acquisition
            new_x, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            # Evaluate
            new_y = func(new_x.detach().numpy().flatten())
            X_train = torch.cat([X_train, new_x])
            Y_train = torch.cat([Y_train, torch.tensor([[new_y]], dtype=torch.double)])
            curr_budget += 1

        best_idx = torch.argmin(Y_train)
        return float(Y_train[best_idx]), X_train[best_idx].numpy()