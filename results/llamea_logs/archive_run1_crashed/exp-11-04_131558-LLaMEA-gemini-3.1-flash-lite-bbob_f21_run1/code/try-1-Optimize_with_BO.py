import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean

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
        
        X = torch.tensor(self.initial_points, dtype=torch.double)
        Y = torch.tensor([func(x) for x in X.numpy()], dtype=torch.double).view(-1, 1)
        
        if len(X) > self.budget:
            raise Exception("Initial points exceed budget")

        while len(X) < self.budget:
            # Model definition
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, Y, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Acquisition Function
            acq = qLogExpectedImprovement(model, best_f=Y.min())
            
            # Optimization of acquisition function
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            # Evaluate
            new_y = torch.tensor([[func(new_x.numpy().flatten())]], dtype=torch.double)
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, new_y])

        best_idx = torch.argmin(Y)
        return float(Y[best_idx]), X[best_idx].numpy()