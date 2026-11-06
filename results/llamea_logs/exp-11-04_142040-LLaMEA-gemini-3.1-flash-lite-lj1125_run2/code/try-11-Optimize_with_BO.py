import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        self.f_opt = float('inf')
        self.x_opt = None
        self.call_count = 0

    def __call__(self, func):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        bounds = torch.stack([lb, ub])

        xs = []
        fs = []

        def safe_eval(x):
            if self.call_count >= self.budget:
                raise Exception("Budget exceeded")
            val = func(x)
            self.call_count += 1
            if val < self.f_opt:
                self.f_opt = val
                self.x_opt = x
            return val

        for p in self.initial_points:
            fs.append(safe_eval(p))
            xs.append(p)

        while self.call_count < self.budget:
            train_x = torch.tensor(np.array(xs), dtype=torch.double)
            train_y = torch.tensor(np.array(fs), dtype=torch.double).view(-1, 1)
            
            # Apply log transformation to handle wide orders of magnitude
            # Add small epsilon to ensure positivity
            y_log = torch.log(train_y - train_y.min() + 1e-9)
            
            model = SingleTaskGP(
                train_x, 
                y_log, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_x,
                prune_baseline=True
            )
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            new_x_np = new_x.detach().numpy().flatten()
            fs.append(safe_eval(new_x_np))
            xs.append(new_x_np)

        return self.f_opt, self.x_opt