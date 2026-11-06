import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        X_data = torch.tensor(self.initial_points, dtype=torch.double)
        y_data = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        curr_budget = len(X_data)
        if curr_budget > self.budget:
            raise ValueError("Initial points exceed budget.")

        best_f = y_data.min()
        best_x = X_data[y_data.argmin()]

        while curr_budget < self.budget:
            # Transform objective to handle flat plateaus and deep wells
            y_train = (y_data - y_data.mean()) / (y_data.std() + 1e-9)
            
            # GP model with Matern kernel for flexibility
            model = SingleTaskGP(X_data, y_train, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # qLogNoisyExpectedImprovement to manage scale and explore
            acq_func = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_data,
                prune_baseline=True
            )

            new_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )

            new_y = func(new_x.detach().numpy().flatten())
            X_data = torch.cat([X_data, new_x])
            y_data = torch.cat([y_data, torch.tensor([[new_y]], dtype=torch.double)])
            
            if new_y < best_f:
                best_f = new_y
                best_x = new_x.detach().numpy().flatten()
            
            curr_budget += 1

        return float(best_f), best_x