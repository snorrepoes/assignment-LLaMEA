import torch
import numpy as np
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
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
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        bounds = torch.stack([lb, ub])
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        tr_length = 0.5 * (ub - lb)
        length_min, length_max = 0.01 * (ub - lb), 1.0 * (ub - lb)
        success_counter, failure_counter = 0, 0
        
        while num_calls < self.budget:
            best_idx = torch.argmin(train_y)
            best_val = train_y[best_idx]
            
            # Model with Matern 2.5 ARD kernel
            model = SingleTaskGP(
                train_x, train_y, 
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)),
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define TR
            center = train_x[best_idx]
            tr_lb = torch.clamp(center - tr_length / 2, lb, ub)
            tr_ub = torch.clamp(center + tr_length / 2, lb, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            # Use qLogExpectedImprovement for robust optimization
            acq = qLogExpectedImprovement(model, best_f=best_val)
            
            # Thompson Sampling for exploration, local optimization for exploitation
            candidate, _ = optimize_acqf(
                acq, bounds=tr_bounds, q=1, num_restarts=15, raw_samples=256
            )

            new_x = candidate.detach()
            new_y = func(new_x.numpy().flatten())
            
            # Update TR
            if new_y < train_y.min():
                success_counter += 1
                failure_counter = 0
                if success_counter >= 3:
                    tr_length = torch.clamp(tr_length * 1.2, max=length_max)
                    success_counter = 0
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= 3:
                    tr_length = torch.clamp(tr_length / 1.5, min=length_min)
                    failure_counter = 0
                    if torch.all(tr_length <= length_min + 1e-4):
                        tr_length = 0.5 * (ub - lb)

            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()