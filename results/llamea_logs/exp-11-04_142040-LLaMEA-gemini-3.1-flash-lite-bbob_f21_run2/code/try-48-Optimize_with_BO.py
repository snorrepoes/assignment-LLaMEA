import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.generation import MaxPosteriorSampling

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
        length = 1.6
        min_length = 0.05
        max_length = 2.5
        
        while num_calls < self.budget:
            train_y_min = train_y.min()
            
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1),
                covar_module=covar_module
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_x = train_x[train_y.argmin()]
            
            # Alternate between local NEI and Thompson Sampling for global diversity
            if num_calls % 3 == 0:
                sampler = MaxPosteriorSampling(model=model, replacement=False)
                candidate = sampler(torch.rand(1000, self.dim) * (ub - lb) + lb, num_samples=1)
            else:
                tr_lb = torch.clamp(best_x - length, lb, ub)
                tr_ub = torch.clamp(best_x + length, lb, ub)
                acq = qNoisyExpectedImprovement(model, X_baseline=train_x)
                candidate, _ = optimize_acqf(
                    acq_function=acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, 
                    num_restarts=15, raw_samples=512
                )

            new_x = candidate.detach().flatten().numpy()
            new_y = func(new_x)
            
            if new_y < train_y_min:
                length = min(length * 1.1, max_length)
            else:
                length = max(length * 0.8, min_length)
            
            train_x = torch.cat([train_x, candidate.detach()])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()