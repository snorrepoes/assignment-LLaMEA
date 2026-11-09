import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

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
        bounds = torch.stack([lb, ub])
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        
        # Hyperparameters for Trust Region
        length = 1.6
        min_length = 0.01
        max_length = 1.6
        success_counter = 0
        failure_counter = 0
        success_tol = 3
        failure_tol = 3

        while num_calls < self.budget:
            train_y_min = train_y.min()
            
            # Model fitting
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1),
                covar_module=covar_module
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define Trust Region bounds centered on current best
            best_x = train_x[train_y.argmin()]
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            acq = qNoisyExpectedImprovement(model, X_baseline=train_x)

            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=tr_bounds,
                q=1,
                num_restarts=10,
                raw_samples=256,
            )

            new_x = candidate.detach().flatten().numpy()
            new_y = func(new_x)
            
            # Update TR state
            if new_y < train_y_min:
                success_counter += 1
                failure_counter = 0
                if success_counter >= success_tol:
                    length = min(2.0 * length, max_length)
                    success_counter = 0
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= failure_tol:
                    length /= 2.0
                    failure_counter = 0
            
            train_x = torch.cat([train_x, candidate.detach()])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1
            
            if length < min_length:
                length = max_length

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()