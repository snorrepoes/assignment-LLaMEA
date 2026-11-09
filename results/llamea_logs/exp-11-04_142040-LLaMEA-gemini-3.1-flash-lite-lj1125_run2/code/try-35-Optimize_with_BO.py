import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
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
        
        train_x = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        train_y = [float(func(x)) for x in self.initial_points]

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Use log-transformation for high-dynamic range, adding offset to handle values near 0
            Y_trans = -torch.log(Y - Y.min() + 1e-9)
            
            # Model with Matérn 5/2 kernel and scaled data
            model = SingleTaskGP(X, Y_trans, likelihood=GaussianLikelihood())
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Local search focus: Identify the current best point
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Adaptive Trust Region size: Start wide and shrink to focus on the narrow well
            # Based on success/failure to find better points
            tr_size = 2.0 * (0.9 ** (len(train_x) / 10))
            tr_lb = torch.clamp(best_x - tr_size, lb, ub)
            tr_ub = torch.clamp(best_x + tr_size, lb, ub)

            # Use qNEI for global-local exploration balance
            acq_func = qNoisyExpectedImprovement(model, X_baseline=X)
            
            # Optimize within TR to ensure we don't get stuck on the plateau
            new_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_val = float(func(new_x.detach().numpy()[0]))
            train_x.append(new_x.squeeze(0))
            train_y.append(new_val)

        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()