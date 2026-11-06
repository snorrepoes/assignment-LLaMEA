import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition import PosteriorMean

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
        train_y = torch.tensor([func(x) for x in train_x], dtype=torch.double).view(-1, 1)
        
        num_calls = len(train_x)
        length = 0.5
        success_cnt, failure_cnt = 0, 0
        
        while num_calls < self.budget:
            train_x_norm = normalize(train_x, bounds)
            
            # Using GP with Matern 2.5 and ARD
            model = SingleTaskGP(train_x_norm, train_y, 
                                 covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define Trust Region around current best
            best_idx = torch.argmin(train_y)
            center = train_x_norm[best_idx]
            tr_lb = (center - length / 2).clamp(0, 1)
            tr_ub = (center + length / 2).clamp(0, 1)

            # Thompson Sampling for global exploration + exploitation
            with torch.no_grad():
                posterior = model.posterior(torch.rand(1000, self.dim))
                samples = posterior.rsample()
                # Use Thompson sampling: minimize the sampled surface
                acq = lambda X: -model.posterior(X).rsample()
                
            candidate_norm, _ = optimize_acqf(
                acq_function=lambda X: -model.posterior(X).rsample().squeeze(-1),
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1, num_restarts=10, raw_samples=128,
            )

            new_x = unnormalize(candidate_norm, bounds).detach().squeeze(0).numpy()
            new_y = func(new_x)
            
            # Update TR logic
            if new_y < train_y.min():
                success_cnt += 1
                failure_cnt = 0
            else:
                failure_cnt += 1
                success_cnt = 0
            
            if success_cnt >= 3:
                length = min(length * 2.0, 2.0)
                success_cnt = 0
            elif failure_cnt >= 5:
                length /= 2.0
                failure_cnt = 0
            
            train_x = torch.cat([train_x, torch.tensor([new_x], dtype=torch.double)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1
            if length < 1e-4: break

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()