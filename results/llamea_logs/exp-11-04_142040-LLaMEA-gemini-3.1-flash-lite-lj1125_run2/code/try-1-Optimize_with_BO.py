import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
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
        self.lb = -5.0
        self.ub = 5.0
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim], dtype=torch.double)
        
        X = torch.tensor(self.initial_points, dtype=torch.double)
        Y = []
        for x in X:
            Y.append(func(x.numpy()))
        Y = torch.tensor(Y, dtype=torch.double).unsqueeze(-1)
        
        count = len(X)
        if count > self.budget:
            raise ValueError("Initial points exceed budget")

        while count < self.budget:
            # Transform Y to handle plateaus/deep wells
            # Log transform helps emphasize small differences in deep regions
            y_min = Y.min()
            train_Y = torch.log(Y - y_min + 1e-9)
            
            # Model with Matern Kernel
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, train_Y, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Acquisition: Expected Improvement
            EI = ExpectedImprovement(model, best_f=train_Y.min())
            
            # Optimize acquisition function
            new_x, _ = optimize_acqf(
                EI, bounds=bounds, q=1, num_restarts=10, raw_samples=512
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, torch.tensor([[new_y]], dtype=torch.double)])
            count += 1
            
        best_idx = torch.argmin(Y)
        return Y[best_idx].item(), X[best_idx].numpy()