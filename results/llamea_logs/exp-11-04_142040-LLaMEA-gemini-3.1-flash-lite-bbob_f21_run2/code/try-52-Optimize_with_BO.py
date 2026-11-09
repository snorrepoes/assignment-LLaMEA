import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.utils.transforms import normalize, unnormalize

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.double)
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in train_x], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        length = 0.8
        min_length, max_length = 0.01, 1.6
        success_count, failure_count = 0, 0
        
        while num_calls < self.budget:
            train_x_norm = normalize(train_x, bounds)
            
            # Use Matern 2.5 kernel with ARD for better handling of multimodal landscapes
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x_norm, train_y, likelihood=GaussianLikelihood(), covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq_func = qNoisyExpectedImprovement(model, X_baseline=train_x_norm, prune_baseline=True)

            # Center TR on current best
            best_idx = torch.argmin(train_y)
            center = train_x_norm[best_idx]
            tr_lb = torch.clamp(center - length / 2, 0.0, 1.0)
            tr_ub = torch.clamp(center + length / 2, 0.0, 1.0)

            candidate_norm, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1, num_restarts=10, raw_samples=512,
            )

            new_x = unnormalize(candidate_norm, bounds).detach().squeeze(0).numpy()
            new_y = func(new_x)
            num_calls += 1
            
            # Trust region update logic
            if new_y < torch.min(train_y):
                success_count += 1
                failure_count = 0
            else:
                failure_count += 1
                success_count = 0
            
            if success_count >= 3:
                length = min(2.0 * length, max_length)
                success_count = 0
            elif failure_count >= 3:
                length /= 2.0
                failure_count = 0
            
            train_x = torch.cat([train_x, torch.tensor([new_x], dtype=torch.double)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            
            if length < min_length:
                length = 0.8 

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()