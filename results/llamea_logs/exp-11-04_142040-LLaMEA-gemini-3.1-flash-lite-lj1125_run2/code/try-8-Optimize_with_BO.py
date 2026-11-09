import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior

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
            if len(train_x) >= self.budget: break
            train_x.append(torch.tensor(x, dtype=torch.double))
            train_y.append(func(x))

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Outcome transformation: Log-transform for heavy-tailed deep wells
            Y_trans = -torch.log(Y - Y.min() + 1e-12)
            
            # GP model with Matern 5/2 kernel
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_trans, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define TR-like bounds based on current best
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Use a tightening trust region to focus on the narrow well
            tr_size = 2.0 * (0.95 ** (len(train_x) / 5)) 
            tr_lb = torch.clamp(best_x - tr_size, lb, ub)
            tr_ub = torch.clamp(best_x + tr_size, lb, ub)

            qNEI = qNoisyExpectedImprovement(model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=15,
                raw_samples=512,
            )
            
            new_val = func(new_x.detach().numpy()[0])
            train_x.append(new_x.squeeze(0))
            train_y.append(new_val)

        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()