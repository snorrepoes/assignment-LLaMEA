import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
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
        train_y = [func(x) for x in self.initial_points]
        
        length = 2.0
        length_min, length_max = 0.005, 4.0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y_raw = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            # Log transform to handle high-dynamic range
            Y = torch.log(Y_raw - Y_raw.min() + 1e-9)
            
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                X, Y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1),
                covar_module=kernel
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = torch.argmin(Y_raw)
            best_x = X[best_idx]
            
            # Trust region boundaries
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qLogNoisyExpectedImprovement(model, X_baseline=X, prune_baseline=True)
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = func(new_x_val.numpy())
            
            # Adaptive trust region logic
            if new_f < train_y[best_idx]:
                length = min(length * 1.2, length_max)
            else:
                length = max(length * 0.8, length_min)
                # Restart length if TR becomes too small
                if length <= length_min: length = 1.0
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        train_y_arr = np.array(train_y)
        best_idx = np.argmin(train_y_arr)
        return float(train_y_arr[best_idx]), train_x[best_idx].numpy()