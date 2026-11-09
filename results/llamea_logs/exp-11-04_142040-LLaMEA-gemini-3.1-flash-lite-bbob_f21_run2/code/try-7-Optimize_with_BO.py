import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        X_list = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        Y_list = [torch.tensor([func(x)], dtype=torch.double) for x in self.initial_points]
        
        if len(X_list) > self.budget:
            raise ValueError("Initial points exceed budget.")

        best_x = X_list[np.argmin([y.item() for y in Y_list])]
        best_f = min([y.item() for y in Y_list])

        while len(X_list) < self.budget:
            train_X = torch.stack(X_list)
            train_Y = torch.stack(Y_list)
            
            # Using Matern 2.5 kernel for better handling of non-smooth multimodal landscapes
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_X, train_Y, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Using Expected Improvement (EI) for efficient exploitation and exploration
            acq_func = ExpectedImprovement(model, best_f=best_f.item(), maximize=False)
            
            bounds = torch.stack([lb, ub])
            candidates, _ = optimize_acqf(
                acq_func, bounds=bounds, 
                q=1, num_restarts=10, raw_samples=512
            )
            
            new_x = candidates.detach().numpy()[0]
            new_y = func(new_x)
            
            X_list.append(torch.tensor(new_x, dtype=torch.double))
            Y_list.append(torch.tensor([new_y], dtype=torch.double))
            
            if new_y < best_f:
                best_f = new_y
                best_x = new_x

        return best_f, best_x.numpy() if isinstance(best_x, torch.Tensor) else best_x