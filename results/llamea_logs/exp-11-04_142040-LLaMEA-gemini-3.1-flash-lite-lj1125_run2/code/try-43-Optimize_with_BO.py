import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan

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
        train_y = [func(x) for x in self.initial_points]

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double)
            
            # Outcome transformation: Log-transform for high dynamic range
            y_min = Y.min()
            Y_trans = torch.log(Y - y_min + 1e-9).unsqueeze(-1)
            
            # GP model with ARD kernel
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y_trans, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define dynamic Trust Region centered at current best
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            # Trust region size shrinks as we iterate to refine local search
            tr_length = 2.0 * (0.8 ** (len(train_x) / 10))
            tr_lb = torch.clamp(best_x - tr_length, lb, ub)
            tr_ub = torch.clamp(best_x + tr_length, lb, ub)

            qLogEI = qLogExpectedImprovement(model, best_f=Y_trans.min())
            
            new_x, _ = optimize_acqf(
                acq_function=qLogEI,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_val = func(new_x.detach().numpy()[0])
            train_x.append(new_x.squeeze(0))
            train_y.append(new_val)

        train_y_arr = np.array(train_y)
        best_idx = np.argmin(train_y_arr)
        return float(train_y_arr[best_idx]), train_x[best_idx].numpy()