import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel

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
        
        best_f = min(train_y)
        length = 0.5
        length_min, length_max = 0.01, 2.0
        success_counter, failure_counter = 0, 0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Log-transform to handle high dynamic range; avoid log(0)
            Y_raw = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            Y = torch.log(Y_raw - Y_raw.min() + 1e-12)
            
            model = SingleTaskGP(
                X, Y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = np.argmin(train_y)
            best_x = X[best_idx]
            
            # Trust region constraints
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qExpectedImprovement(model, best_f=Y[best_idx])
            
            new_x, _ = optimize_acqf(
                acq_function=acq, bounds=torch.stack([tr_lb, tr_ub]),
                q=1, num_restarts=10, raw_samples=512
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = float(func(new_x_val.numpy()))
            
            if new_f < best_f:
                best_f = new_f
                success_counter += 1
                failure_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
            
            if success_counter >= 3:
                length = min(length * 2.0, length_max)
                success_counter = 0
            elif failure_counter >= 5:
                length = max(length * 0.5, length_min)
                failure_counter = 0
                if length <= length_min: # Restart logic
                    length = 1.0
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        final_y = np.array(train_y)
        best_idx = np.argmin(final_y)
        return float(final_y[best_idx]), train_x[best_idx].numpy()