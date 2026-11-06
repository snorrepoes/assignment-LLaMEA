import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
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
        
        # TuRBO state
        tr_length = 1.0 * (ub - lb)
        length_min, length_max = 0.05 * (ub - lb), 1.6 * (ub - lb)
        success_counter, failure_counter = 0, 0
        
        while num_calls < self.budget:
            train_y_min = train_y.min()
            best_idx = torch.argmin(train_y)
            center = train_x[best_idx]
            
            # Create local bounds
            tr_lb = torch.clamp(center - tr_length / 2, lb, ub)
            tr_ub = torch.clamp(center + tr_length / 2, lb, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            # Model fitting
            model = SingleTaskGP(
                train_x, train_y, 
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)),
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Alternate between Local TR and Global NEI to handle multimodal landscape
            if num_calls % 3 == 0:
                acq = qLogNoisyExpectedImprovement(model, X_baseline=train_x)
                candidate, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
            else:
                acq = qLogNoisyExpectedImprovement(model, X_baseline=train_x)
                candidate, _ = optimize_acqf(acq, bounds=tr_bounds, q=1, num_restarts=10, raw_samples=256)

            new_x = candidate.detach()
            new_y = func(new_x.numpy().flatten())
            
            # TR update
            if new_y < train_y_min:
                success_counter += 1
                failure_counter = 0
                if success_counter >= 3:
                    tr_length = torch.clamp(tr_length * 1.5, max=length_max)
                    success_counter = 0
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= 3:
                    tr_length = torch.clamp(tr_length / 1.5, min=length_min)
                    failure_counter = 0

            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()