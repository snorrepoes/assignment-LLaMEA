import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
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
        
        num_calls = len(train_x)
        
        # State variables for TuRBO-like trust region
        best_x = train_x[torch.argmin(train_y)]
        length = 0.8
        
        while num_calls < self.budget:
            train_x_norm = normalize(train_x, bounds)
            
            likelihood = GaussianLikelihood()
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x_norm, train_y, likelihood=likelihood, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Using qLogExpectedImprovement for better numerical stability and search
            acq_func = qLogExpectedImprovement(model, best_f=train_y.min())
            
            # Constrain search to a local hypercube (Trust Region)
            best_x_norm = normalize(best_x.unsqueeze(0), bounds).squeeze(0)
            tr_lb = (best_x_norm - length / 2).clamp(0, 1)
            tr_ub = (best_x_norm + length / 2).clamp(0, 1)
            
            candidate_norm, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1, num_restarts=10, raw_samples=256,
            )
            
            new_x = unnormalize(candidate_norm, bounds).detach().squeeze(0).numpy()
            new_y = func(new_x)
            
            train_x = torch.cat([train_x, torch.tensor([new_x], dtype=torch.double)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            
            if new_y < train_y[:-1].min():
                best_x = torch.tensor(new_x, dtype=torch.double)
                length = min(length * 1.1, 2.0)
            else:
                length = length * 0.95
                
            num_calls += 1
            if length < 1e-3: length = 0.5 # Restart region if too small
            
        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()