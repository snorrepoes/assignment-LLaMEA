import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
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
        self.seed = random_seed
        self.f_opt = float('inf')
        self.x_opt = None
        self.call_count = 0

    def __call__(self, func):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        bounds = torch.stack([lb, ub])

        xs = []
        fs = []

        def safe_eval(x):
            if self.call_count >= self.budget:
                raise Exception("Budget exceeded")
            val = func(x)
            self.call_count += 1
            if val < self.f_opt:
                self.f_opt = val
                self.x_opt = x
            return val

        # Initial points
        for p in self.initial_points:
            fs.append(safe_eval(p))
            xs.append(p)

        # Optimization loop
        while self.call_count < self.budget:
            train_x = torch.tensor(np.array(xs), dtype=torch.double)
            train_y = torch.tensor(np.array(fs), dtype=torch.double).view(-1, 1)
            
            # Standardize output for robust GP modeling
            train_y_std = (train_y - train_y.mean()) / (train_y.std() + 1e-9)

            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x, train_y_std, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            fit_gpytorch_mll(mll)

            acq = qLogExpectedImprovement(model, best_f=train_y_std.min())
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            new_x_np = new_x.detach().numpy().flatten()
            fs.append(safe_eval(new_x_np))
            xs.append(new_x_np)

        return self.f_opt, self.x_opt