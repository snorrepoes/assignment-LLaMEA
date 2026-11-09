import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

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
        train_y = [func(x) for x in self.initial_points]
        
        # Hyperparameters for TuRBO-1
        length = 1.6
        min_length = 0.01
        max_length = 1.6
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y_raw = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            
            # Use log-transformed Y to suppress flat plateaus and highlight the well
            # Shift to ensure positivity for log-transform
            y_min = Y_raw.min()
            Y = torch.log(Y_raw - y_min + 1e-12)
            
            model = SingleTaskGP(
                X, Y,
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            model.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = torch.argmin(Y_raw)
            best_x = X[best_idx]
            
            # Define trust region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qLogExpectedImprovement(model, best_f=Y.min())
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=15,
                raw_samples=512,
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = func(new_x_val.numpy())
            
            # Update TR length based on success
            if new_f < train_y[best_idx]:
                length = min(length * 1.1, max_length)
            else:
                length = max(length * 0.5, min_length)
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        idx = np.argmin(train_y)
        return float(train_y[idx]), train_x[idx].numpy()