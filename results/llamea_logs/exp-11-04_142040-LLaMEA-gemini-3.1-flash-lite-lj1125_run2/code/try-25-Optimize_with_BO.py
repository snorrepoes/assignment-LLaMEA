import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan

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
        
        best_val = min(train_y)
        best_x = train_x[np.argmin(train_y)]
        
        # State variables for TuRBO
        length = 1.6
        length_min, length_max = 0.01, 2.0
        success_counter, failure_counter = 0, 0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            
            # Normalize target to handle high dynamic range and log-space focus
            Y_norm = (Y - Y.mean()) / Y.std()
            
            likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4))
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_norm, covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Local Trust Region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qLogNoisyExpectedImprovement(model, X_baseline=X, prune_baseline=True)
            new_x, _ = optimize_acqf(acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, 
                                     num_restarts=10, raw_samples=128)
            
            new_x = new_x.detach().flatten()
            new_f = float(func(new_x.numpy()))
            
            train_x.append(new_x)
            train_y.append(new_f)
            
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                success_counter += 1
                failure_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
            
            # Adjust trust region size
            if success_counter >= 3:
                length = min(length * 2.0, length_max)
                success_counter = 0
            elif failure_counter >= 5:
                length = max(length / 2.0, length_min)
                failure_counter = 0
                
        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()