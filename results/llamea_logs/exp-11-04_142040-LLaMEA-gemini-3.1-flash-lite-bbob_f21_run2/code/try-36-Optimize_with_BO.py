import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qLogNoisyExpectedImprovement
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
        length = 1.0
        min_length = 0.05
        max_length = 2.0
        success_counter = 0
        failure_counter = 0
        
        while num_calls < self.budget:
            # Model fitting with standardized outputs for better stability
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1),
                covar_module=covar_module
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define Trust Region centered on current best
            best_val, best_idx = torch.min(train_y, dim=0)
            best_x = train_x[best_idx].squeeze()
            
            tr_lb = torch.max(best_x - length, lb)
            tr_ub = torch.min(best_x + length, ub)
            
            # Using Log-scale acquisition for better tail behavior
            acq = qLogNoisyExpectedImprovement(model, X_baseline=train_x, prune_baseline=True)

            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=512,
            )

            new_x = candidate.detach().flatten()
            new_y = torch.tensor([[func(new_x.numpy())]], dtype=torch.double)
            
            # Update TR state
            if new_y < best_val:
                success_counter += 1
                failure_counter = 0
                if success_counter >= 2:
                    length = min(length * 1.2, max_length)
                    success_counter = 0
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= 3:
                    length = max(length / 1.5, min_length)
                    failure_counter = 0
            
            train_x = torch.cat([train_x, new_x.unsqueeze(0)])
            train_y = torch.cat([train_y, new_y])
            num_calls += 1
            
            # Reset length if too small to escape local minima
            if length <= min_length and failure_counter >= 3:
                length = 1.0

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()