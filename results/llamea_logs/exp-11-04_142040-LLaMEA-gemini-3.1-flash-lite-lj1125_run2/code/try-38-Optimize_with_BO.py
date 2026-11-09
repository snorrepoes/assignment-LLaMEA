import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

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
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Log transform to handle dynamic range (minimizing, so -log(y - min + eps))
            y_min = Y.min()
            Y_trans = -torch.log(Y - y_min + 1e-9)
            
            # Model with Matern 5/2 kernel and noise learning
            model = SingleTaskGP(X, Y_trans, likelihood=GaussianLikelihood())
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # TuRBO-style local search: focus on current best
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Decay trust region size from 2.0 to 0.1 over the budget
            progress = len(train_x) / self.budget
            tr_size = 2.0 * (0.05 ** progress)
            
            tr_lb = torch.clamp(best_x - tr_size, lb, ub)
            tr_ub = torch.clamp(best_x + tr_size, lb, ub)

            # Optimize EI within the trust region
            ei = qExpectedImprovement(model, best_f=Y_trans.max())
            new_x, _ = optimize_acqf(
                acq_function=ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=10,
                raw_samples=256,
            )
            
            val = func(new_x.detach().numpy()[0])
            train_x.append(new_x.squeeze(0))
            train_y.append(float(val))

        train_y_arr = np.array(train_y)
        best_idx = np.argmin(train_y_arr)
        return float(train_y_arr[best_idx]), train_x[best_idx].numpy()