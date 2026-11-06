import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf

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
        
        length = 2.0
        length_min, length_max = 0.5, 4.0
        success_counter, failure_counter = 0, 0
        success_tol, failure_tol = 3, 3

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Log transform Y to handle high dynamic range: Y = -log(y - min + eps)
            y_tensor = torch.tensor(train_y, dtype=torch.double)
            y_min = y_tensor.min()
            Y_trans = -torch.log(y_tensor - y_min + 1e-9).unsqueeze(-1)
            
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_trans, covar_module=covar_module, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acqf = qLogNoisyExpectedImprovement(model, X_baseline=X, sampler=None)
            
            new_x, _ = optimize_acqf(
                acqf, bounds=torch.stack([tr_lb, tr_ub]), q=1, num_restarts=10, raw_samples=128
            )
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
                
            if success_counter >= success_tol:
                length = min(length * 1.5, length_max)
                success_counter = 0
            elif failure_counter >= failure_tol:
                length = max(length / 1.5, length_min)
                failure_counter = 0
        
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()