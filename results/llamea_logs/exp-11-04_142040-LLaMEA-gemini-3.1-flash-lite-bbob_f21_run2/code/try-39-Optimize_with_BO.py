import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

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
        tr_length = 0.5 * (ub - lb)
        length_min, length_max = 0.05 * (ub - lb), 1.0 * (ub - lb)
        success_counter, failure_counter = 0, 0
        
        while num_calls < self.budget:
            best_idx = torch.argmin(train_y)
            train_y_min = train_y[best_idx]
            
            # Use centered TR around best point
            center = train_x[best_idx]
            tr_lb = torch.max(center - tr_length / 2, lb)
            tr_ub = torch.min(center + tr_length / 2, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            model = SingleTaskGP(
                train_x, train_y, 
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)),
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Use UCB for balancing exploration/exploitation within TR
            acq = qUpperConfidenceBound(model, beta=2.0)
            candidate, _ = optimize_acqf(acq, bounds=tr_bounds, q=1, num_restarts=15, raw_samples=512)

            new_x = candidate.detach()
            new_y = torch.tensor([[func(new_x.numpy().flatten())]], dtype=torch.double)
            
            # Update TR logic
            if new_y < train_y_min:
                success_counter += 1
                failure_counter = 0
                if success_counter >= 3:
                    tr_length = torch.min(tr_length * 1.5, length_max)
                    success_counter = 0
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= 4:
                    tr_length = torch.max(tr_length / 1.5, length_min)
                    failure_counter = 0

            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])
            num_calls += 1

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()