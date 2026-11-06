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
        
        # Initial DoE
        for x in self.initial_points:
            f = func(x)
            train_x.append(x)
            train_y.append(f)
        
        train_x = torch.tensor(np.array(train_x), dtype=torch.double)
        train_y = torch.tensor(np.array(train_y), dtype=torch.double).unsqueeze(-1)
        
        while len(train_x) < self.budget:
            # Log-transform targets to handle dynamic range and plateaus
            y_min = train_y.min()
            target_y = torch.log(train_y - y_min + 1e-9)
            
            # Setup GP
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x, target_y, likelihood=GaussianLikelihood(), covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Acquisition Function
            acq_func = qLogExpectedImprovement(model, best_f=target_y.min())
            
            # Optimize Acquisition
            new_x, _ = optimize_acqf(
                acq_func, bounds=torch.stack([lb, ub]), q=1, num_restarts=10, raw_samples=512
            )
            
            # Evaluate
            new_x = new_x.detach().numpy()[0]
            new_f = func(new_x)
            
            train_x = torch.cat([train_x, torch.tensor(new_x, dtype=torch.double).unsqueeze(0)])
            train_y = torch.cat([train_y, torch.tensor(new_f, dtype=torch.double).unsqueeze(0).unsqueeze(0)])
            
        best_idx = torch.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()