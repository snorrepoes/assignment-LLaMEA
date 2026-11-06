import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean

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
        
        train_x = []
        train_y = []

        # Evaluate initial points
        for x in self.initial_points:
            if len(train_x) >= self.budget:
                break
            f = func(x)
            train_x.append(torch.tensor(x, dtype=torch.double))
            train_y.append(f)

        # Optimization loop
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Log transform Y to handle plateaus and deep wells
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            Y_log = -torch.log(Y - Y.min() + 1e-9) 

            # Fit GP
            model = SingleTaskGP(X, Y_log)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Acquisition
            qNEI = qNoisyExpectedImprovement(model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=10,
                raw_samples=200,
            )
            
            new_x = new_x.detach().numpy()[0]
            new_f = func(new_x)
            
            train_x.append(torch.tensor(new_x, dtype=torch.double))
            train_y.append(new_f)

        # Result
        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()