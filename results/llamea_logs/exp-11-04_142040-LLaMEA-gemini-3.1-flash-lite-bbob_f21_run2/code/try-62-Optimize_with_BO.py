import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
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
        bounds_np = np.array([func.bounds.lb, func.bounds.ub])
        bounds = torch.tensor(bounds_np, dtype=torch.double)
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in train_x], dtype=torch.double).view(-1, 1)
        
        # We need to minimize, so we negate y for standard BoTorch acquisition functions
        # or use the maximize=False flag correctly.
        num_calls = len(train_x)
        
        while num_calls < self.budget:
            # Normalize inputs to [0, 1]
            train_x_norm = normalize(train_x, bounds)
            
            # Use Matern 5/2 with ARD for better adaptation to different input scales
            likelihood = GaussianLikelihood()
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x_norm, -train_y, likelihood=likelihood, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with torch.no_grad():
                fit_gpytorch_mll(mll)
            
            # LogExpectedImprovement is more robust numerically
            acq_func = LogExpectedImprovement(model, best_f=torch.max(-train_y), maximize=True)
            
            candidate_norm, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                q=1, 
                num_restarts=10, 
                raw_samples=512,
            )
            
            new_x = unnormalize(candidate_norm, bounds).detach().squeeze(0).numpy()
            new_y = func(new_x)
            
            train_x = torch.cat([train_x, torch.tensor([new_x], dtype=torch.double)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1
            
        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()