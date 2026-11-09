import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.constraints import GreaterThan

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.float64)
        ub = torch.tensor(func.bounds.ub, dtype=torch.float64)
        
        train_x = []
        train_y = []
        
        def evaluate(x):
            if len(train_x) >= self.budget:
                raise Exception("Budget exhausted")
            val = func(x)
            train_x.append(x)
            train_y.append(val)
            return val

        # Initial DoE
        for p in self.initial_points:
            evaluate(p)
            
        while len(train_x) < self.budget:
            X = torch.tensor(np.array(train_x), dtype=torch.float64)
            # Log-transform targets to handle large dynamic range/plateaus
            y_vals = np.array(train_y)
            y_min = y_vals.min()
            y_transformed = torch.tensor(np.log(y_vals - y_min + 1e-9), dtype=torch.float64).unsqueeze(-1)
            
            # Setup GP
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, y_transformed, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Acquisition: qNEI
            acq = qNoisyExpectedImprovement(model, X_baseline=X)
            
            # Optimize acquisition
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=10,
                raw_samples=512
            )
            
            evaluate(new_x.detach().numpy().flatten())
            
        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx]