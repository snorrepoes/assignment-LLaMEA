import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
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
        
        X_list = []
        Y_list = []
        
        def evaluate(x):
            if len(X_list) >= self.budget:
                raise Exception("Evaluation budget exceeded.")
            val = func(x)
            X_list.append(torch.tensor(x, dtype=torch.double))
            Y_list.append(val)
            return val

        # Evaluate initial points
        for x_init in self.initial_points:
            evaluate(x_init)

        # Optimization loop
        while len(X_list) < self.budget:
            train_X = torch.stack(X_list)
            # Log transform Y to handle high dynamic range/flat plateaus
            train_Y = torch.tensor(Y_list, dtype=torch.double).unsqueeze(-1)
            train_Y_log = torch.log(train_Y - train_Y.min() + 1e-9)
            
            # Standardize
            train_Y_mean, train_Y_std = train_Y_log.mean(), train_Y_log.std()
            train_Y_std = torch.clamp(train_Y_std, min=1e-9)
            train_Y_norm = (train_Y_log - train_Y_mean) / train_Y_std

            # Model
            model = SingleTaskGP(train_X, train_Y_norm)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Acquisition
            acq_func = qLogExpectedImprovement(model, best_f=train_Y_norm.max())
            
            # Optimize acquisition
            candidates, _ = optimize_acqf(
                acq_func, bounds=torch.stack([lb, ub]), q=1, num_restarts=10, raw_samples=512
            )
            
            evaluate(candidates.detach().numpy().flatten())

        best_idx = np.argmin(Y_list)
        return Y_list[best_idx], X_list[best_idx].numpy()