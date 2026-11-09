import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
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
        
        train_x_list = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        train_y_list = [float(func(x)) for x in self.initial_points]
        
        if len(train_x_list) > self.budget:
            raise ValueError("Initial points exceed budget.")

        while len(train_x_list) < self.budget:
            train_x = torch.stack(train_x_list)
            train_y = torch.tensor(train_y_list, dtype=torch.double).unsqueeze(-1)
            
            # Using Normalize and Standardize for robust surrogate training
            model = SingleTaskGP(
                train_x, 
                train_y, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            model.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # qLogNEI is more robust for landscapes with massive dynamic range
            acq_func = qLogNoisyExpectedImprovement(model, X_baseline=train_x)
            
            new_x, _ = optimize_acqf(
                acq_func, 
                bounds=torch.stack([lb, ub]), 
                q=1, 
                num_restarts=20, 
                raw_samples=512
            )
            
            new_x_val = new_x.detach().squeeze()
            new_f_val = float(func(new_x_val.numpy()))
            
            train_x_list.append(new_x_val)
            train_y_list.append(new_f_val)
            
        best_idx = np.argmin(train_y_list)
        return float(train_y_list[best_idx]), train_x_list[best_idx].numpy()