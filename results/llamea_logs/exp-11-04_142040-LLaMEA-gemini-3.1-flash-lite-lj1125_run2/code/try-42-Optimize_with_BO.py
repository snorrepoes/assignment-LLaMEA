import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms.outcome import Standardize
from torch.quasirandom import SobolEngine

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

        for x in self.initial_points:
            train_x.append(torch.tensor(x, dtype=torch.double))
            train_y.append(float(func(x)))

        length = 0.8
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Log-transform to handle flat plateaus and deep wells
            Y_log = torch.log(Y - Y.min() + 1e-9)
            
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_log, covar_module=covar_module, likelihood=GaussianLikelihood())
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Trust region logic: if we don't improve, shrink; if we do, expand slightly
            if len(train_x) > 1 and train_y[-1] < min(train_y[:-1]):
                length = min(2.0, length * 1.1)
            else:
                length = max(0.01, length * 0.95)
            
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)

            acq_func = qLogExpectedImprovement(model, best_f=Y_log.min())
            
            new_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=10,
                raw_samples=256,
            )
            
            new_val = float(func(new_x.detach().numpy()[0]))
            train_x.append(new_x.squeeze(0))
            train_y.append(new_val)

        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()