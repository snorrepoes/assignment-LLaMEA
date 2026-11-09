import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.noisy_ei import qNoisyExpectedImprovement
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
        self.best_f = float('inf')
        self.best_x = None
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        if num_calls > self.budget:
            raise ValueError("Initial points exceed budget.")

        while num_calls < self.budget:
            # Update best found
            best_idx = torch.argmin(train_y)
            self.best_f = train_y[best_idx].item()
            self.best_x = train_x[best_idx].numpy()

            # Define GP model
            likelihood = GaussianLikelihood()
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x, train_y, likelihood=likelihood, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            fit_gpytorch_mll(mll)

            # Define acquisition function
            qNEI = qNoisyExpectedImprovement(model, train_x)

            # Optimize acquisition function
            candidate, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )

            # Evaluate
            new_x = candidate.detach().numpy()[0]
            new_y = func(new_x)
            
            train_x = torch.cat([train_x, candidate.detach()])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1

        # Final update
        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()