import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.eval_count = 0

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        X_list, y_list = [], []
        
        for x in self.initial_points:
            if self.eval_count >= self.budget: break
            X_list.append(torch.tensor(x, dtype=torch.double))
            y_list.append(func(x))
            self.eval_count += 1
            
        while self.eval_count < self.budget:
            train_X = torch.stack(X_list)
            train_y = torch.tensor(y_list, dtype=torch.double).view(-1, 1)
            
            # Log transform y to handle potential wide dynamic range
            train_y_log = torch.log(train_y - train_y.min() + 1e-9)
            
            model = SingleTaskGP(train_X, train_y_log, 
                                 likelihood=GaussianLikelihood(),
                                 covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            acq_func = qNoisyExpectedImprovement(model, train_X)
            
            new_x, _ = optimize_acqf(
                acq_func, bounds=torch.stack([lb, ub]), 
                q=1, num_restarts=10, raw_samples=512
            )
            
            new_x = new_x.detach().numpy().flatten()
            new_y = func(new_x)
            
            X_list.append(torch.tensor(new_x, dtype=torch.double))
            y_list.append(new_y)
            self.eval_count += 1
            
        y_arr = np.array(y_list)
        best_idx = np.argmin(y_arr)
        return y_arr[best_idx], X_list[best_idx].numpy()