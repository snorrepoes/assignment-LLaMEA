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
        length = 0.5 * (ub - lb)
        length_min, length_max = 0.01, 1.6
        failure_counter, success_counter = 0, 0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y_raw = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            # Log transform to handle flat plateaus
            Y_transformed = torch.log(Y_raw - Y_raw.min() + 1e-9)
            
            # Model with Matern kernel for flexibility
            model = SingleTaskGP(
                X, Y_transformed,
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = torch.argmin(Y_raw)
            best_x = X[best_idx]
            
            # Trust region constraints
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qLogExpectedImprovement(model, best_f=Y_transformed.min())
            
            # Optimized acquisition search
            new_x, _ = optimize_acqf(
                acq_function=acq, bounds=torch.stack([tr_lb, tr_ub]),
                q=1, num_restarts=15, raw_samples=512
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = func(new_x_val.numpy())
            
            # Update TR length
            if new_f < Y_raw[best_idx]:
                success_counter += 1
                failure_counter = 0
                if success_counter >= 3:
                    length = torch.clamp(length * 2.0, length_min, length_max)
                    success_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
                if failure_counter >= 5:
                    length = torch.clamp(length * 0.5, length_min, length_max)
                    failure_counter = 0
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        final_y = np.array(train_y)
        best_idx = np.argmin(final_y)
        return float(final_y[best_idx]), train_x[best_idx].numpy()