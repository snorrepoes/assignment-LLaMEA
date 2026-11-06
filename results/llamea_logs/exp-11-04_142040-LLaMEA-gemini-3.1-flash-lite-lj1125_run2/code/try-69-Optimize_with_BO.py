import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
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
        
        # Hyperparameters for TuRBO-1
        length = 0.5  # Initial trust region half-side length
        length_min, length_max = 0.005, 2.0
        success_counter = 0
        failure_counter = 0
        success_threshold = 3
        failure_threshold = 3

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y_raw = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            
            # Log transform to compress the plateau and emphasize the well
            # Shift by min and add small epsilon to ensure positivity
            y_min = Y_raw.min()
            Y = torch.log(Y_raw - y_min + 1e-12)
            
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
            
            # Constrain to Trust Region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            # Use qLogNEI for exploration
            acq = qLogNoisyExpectedImprovement(model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = float(func(new_x_val.numpy()))
            
            # Update counters and length
            if new_f < min(train_y):
                success_counter += 1
                failure_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
                
            if success_counter >= success_threshold:
                length = min(length * 2.0, length_max)
                success_counter = 0
            elif failure_counter >= failure_threshold:
                length = max(length * 0.5, length_min)
                failure_counter = 0
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        train_y_arr = np.array(train_y)
        best_idx = np.argmin(train_y_arr)
        return float(train_y_arr[best_idx]), train_x[best_idx].numpy()