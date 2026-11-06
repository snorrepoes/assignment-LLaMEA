import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

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
        
        # Hyperparameters for TuRBO-m
        length = 1.6
        min_length = 0.05
        max_length = 1.6
        failure_counter = 0
        failure_tol = 4

        while len(train_x) < self.budget:
            train_y_min = train_y.min()
            
            # Use Standardize and Normalize for robust GP regression
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Trust Region centered on the best observation
            best_x = train_x[train_y.argmin()]
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            # Use qNoisyExpectedImprovement for robust local exploitation
            acq = qNoisyExpectedImprovement(model, X_baseline=train_x)

            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=tr_bounds,
                q=1,
                num_restarts=15,
                raw_samples=512,
            )

            new_x = candidate.detach().flatten().numpy()
            new_y = func(new_x)
            
            # Dynamic TR adjustment
            if new_y < train_y_min:
                failure_counter = 0
                length = min(length * 1.1, max_length)
            else:
                failure_counter += 1
                if failure_counter >= failure_tol:
                    length = max(length / 2.0, min_length)
                    failure_counter = 0
            
            train_x = torch.cat([train_x, candidate.detach()])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()