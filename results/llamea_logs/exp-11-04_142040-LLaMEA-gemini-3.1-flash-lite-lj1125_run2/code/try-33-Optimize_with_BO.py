import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
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
        
        length = 0.5 * (ub - lb)
        length_min = 0.01
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Use a log-transform to handle potential massive range differences
            # Add small epsilon to handle non-positive values if necessary
            Y_trans = torch.log(Y - Y.min() + 1e-12)
            
            model = SingleTaskGP(
                X, Y_trans,
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = np.argmin(train_y)
            best_x = X[best_idx]
            
            # Constrain search to a local trust region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qLogExpectedImprovement(model, best_f=Y_trans.min())
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = float(func(new_x_val.numpy()))
            
            # Trust region logic
            if new_f < train_y[best_idx]:
                length = torch.clamp(length * 1.05, min=length_min)
            else:
                length = torch.clamp(length * 0.9, min=length_min)
                
            train_x.append(new_x_val)
            train_y.append(new_f)

        train_y_arr = np.array(train_y)
        best_idx = np.argmin(train_y_arr)
        return float(train_y_arr[best_idx]), train_x[best_idx].numpy()