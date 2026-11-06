import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
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
        length = 0.8
        min_length = 0.01
        max_length = 1.6
        success_tol = 3
        failure_tol = 5
        success_count = 0
        failure_count = 0
        
        best_x = train_x[torch.argmin(train_y)]
        best_y = train_y.min()

        while num_calls < self.budget:
            train_y_std = (train_y - train_y.mean()) / (train_y.std() + 1e-8)
            model = SingleTaskGP(
                train_x, train_y_std, 
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)),
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Thompson Sampling for global exploration
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            candidate = thompson_sampling(num_samples=1, observation_range=None)
            
            new_y = func(candidate.detach().numpy().flatten())
            num_calls += 1
            
            if new_y < best_y:
                best_y = new_y
                best_x = candidate
                success_count += 1
                failure_count = 0
            else:
                success_count = 0
                failure_count += 1
            
            if success_count >= success_tol:
                length = min(2.0 * length, max_length)
                success_count = 0
            elif failure_count >= failure_tol:
                length /= 2.0
                failure_count = 0
            
            train_x = torch.cat([train_x, candidate])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            
            if length < min_length:
                length = 0.8 # Restart Trust Region

        return best_y, best_x.numpy().flatten()