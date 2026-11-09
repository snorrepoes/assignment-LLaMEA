import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

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
        
        train_x_list = []
        train_y_list = []
        
        # Initial points evaluation
        for x in self.initial_points:
            train_x_list.append(torch.tensor(x, dtype=torch.double))
            train_y_list.append(func(x))
            
        if len(train_x_list) > self.budget:
            raise ValueError("Budget exceeded.")

        # BO loop
        while len(train_x_list) < self.budget:
            train_x = torch.stack(train_x_list)
            # Use log transform to compress the dynamic range of the objective
            train_y = torch.tensor(train_y_list, dtype=torch.double).unsqueeze(-1)
            train_y_transformed = torch.log(train_y - train_y.min() + 1e-12)
            
            model = SingleTaskGP(
                train_X=train_x, 
                train_Y=train_y_transformed,
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            qNEI = qNoisyExpectedImprovement(model, X_baseline=train_x)
            
            new_x, _ = optimize_acqf(
                acq_function=qNEI, 
                bounds=torch.stack([lb, ub]), 
                q=1, 
                num_restarts=20, 
                raw_samples=512
            )
            
            next_x = new_x.detach().flatten().numpy()
            train_x_list.append(torch.tensor(next_x, dtype=torch.double))
            train_y_list.append(func(next_x))
            
        best_idx = np.argmin(train_y_list)
        return float(train_y_list[best_idx]), train_x_list[best_idx].numpy()