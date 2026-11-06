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
        train_y = [float(func(x)) for x in self.initial_points]
        
        # Adaptive Trust Region state
        length = 0.5 * (ub - lb)
        length_min, length_max = 0.01, 1.0
        success_counter = 0
        failure_counter = 0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Log-transform of (Y - min + epsilon) helps capture deep narrow wells
            y_min = Y.min()
            log_y = torch.log(Y - y_min + 1e-9)
            
            # Using Matern 5/2 kernel with automatic relevance determination
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                X, log_y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                covar_module=covar_module
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Trust region constraints
            tr_lb = torch.max(lb, best_x - length)
            tr_ub = torch.min(ub, best_x + length)
            
            acq = qLogExpectedImprovement(model, best_f=log_y.min())
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=512,
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = float(func(new_x_val.numpy()))
            
            # Update trust region logic
            if new_f < train_y[best_idx]:
                success_counter += 1
                failure_counter = 0
                if success_counter >= 3:
                    length = torch.clamp(length * 2.0, max=length_max)
                    success_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
                if failure_counter >= 3:
                    length = torch.clamp(length * 0.5, min=length_min)
                    failure_counter = 0
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        train_y_arr = np.array(train_y)
        best_idx = np.argmin(train_y_arr)
        return float(train_y_arr[best_idx]), train_x[best_idx].numpy()