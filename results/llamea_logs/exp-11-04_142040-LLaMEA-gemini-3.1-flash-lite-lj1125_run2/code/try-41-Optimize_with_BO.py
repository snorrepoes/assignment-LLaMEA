import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms import Normalize, Standardize

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

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            
            # Log transform to handle high-dynamic-range and stabilize GP
            Y_log = torch.log(Y - Y.min() + 1e-9)
            
            model = SingleTaskGP(
                X, Y_log, 
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)),
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # TuRBO-1: Adaptive Trust Region strategy
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            tr_length = 0.5 * (0.8 ** (len(train_x) / 10)) 
            tr_lb = torch.clamp(best_x - tr_length * (ub - lb), lb, ub)
            tr_ub = torch.clamp(best_x + tr_length * (ub - lb), lb, ub)

            # Use UCB with exploration factor beta to jump out of plateaus
            UCB = UpperConfidenceBound(model, beta=2.0)
            
            new_x, _ = optimize_acqf(
                acq_function=UCB,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_val = float(func(new_x.detach().numpy()[0]))
            train_x.append(new_x.squeeze(0))
            train_y.append(new_val)

        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()