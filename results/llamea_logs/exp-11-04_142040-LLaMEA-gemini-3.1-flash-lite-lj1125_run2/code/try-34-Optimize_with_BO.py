import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
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
        
        train_x = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        train_y = [float(func(x)) for x in self.initial_points]

        best_val = min(train_y)
        best_x = train_x[np.argmin(train_y)]
        
        # Hyperparameters for TuRBO
        length = 1.6
        length_min = 0.05
        length_max = 2.0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            # Log transform Y to handle high dynamic range: target minimization
            y_min = min(train_y)
            Y_raw = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            Y_trans = torch.log(Y_raw - y_min + 1e-9)
            
            # GP model with ARD Matern kernel
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_trans, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define Trust Region bounds
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            # Use Log-EI which is more robust for plateaus
            acq = qLogExpectedImprovement(model, best_f=Y_trans.min())
            
            new_x, _ = optimize_acqf(
                acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, 
                num_restarts=20, raw_samples=512
            )
            
            new_x = new_x.detach().flatten()
            new_f = float(func(new_x.numpy()))
            
            train_x.append(new_x)
            train_y.append(new_f)
            
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                length = min(length * 1.1, length_max)
            else:
                length = max(length * 0.5, length_min)

        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()