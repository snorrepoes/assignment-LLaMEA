import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from botorch.models.transforms import Normalize, Standardize

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        train_x = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        train_y = [float(func(x)) for x in self.initial_points]
        
        best_idx = np.argmin(train_y)
        best_x = train_x[best_idx].clone()
        best_val = train_y[best_idx]
        
        length = 1.0
        success_counter = 0
        failure_counter = 0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            # Log transform for high dynamic range handling
            Y_log = torch.log(Y - Y.min() + 1e-9)
            
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_log, covar_module=kernel, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Thompson Sampling for exploration
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            # Generate candidate within trust region
            candidate = torch.rand(1, self.dim, dtype=torch.double) * (tr_ub - tr_lb) + tr_lb
            with torch.no_grad():
                posterior = model.posterior(candidate)
                samples = posterior.rsample(sampler=sampler).squeeze(-1)
            
            new_x = candidate[torch.argmin(samples.mean(dim=0))]
            new_f = float(func(new_x.numpy()))
            
            train_x.append(new_x)
            train_y.append(new_f)
            
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                success_counter += 1
                failure_counter = 0
            else:
                failure_counter += 1
            
            if success_counter >= 3:
                length = min(length * 2.0, 4.0)
                success_counter = 0
            elif failure_counter >= 5:
                length = max(length * 0.5, 0.01)
                failure_counter = 0
                
        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()