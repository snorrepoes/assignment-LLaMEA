import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
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
        train_y = [func(x) for x in self.initial_points]

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Log-transform to handle high dynamic range and flatten the plateau
            Y_min = Y.min()
            Y_trans = torch.log(Y - Y_min + 1e-9)
            
            model = SingleTaskGP(
                X, Y_trans, 
                likelihood=GaussianLikelihood(noise_constraint=None),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Use qLogNoisyExpectedImprovement for better exploration in log-space
            qLogNEI = qLogNoisyExpectedImprovement(model, X_baseline=X)
            
            # Use local optimization restarts to refine the search in promising regions
            new_x, _ = optimize_acqf(
                acq_function=qLogNEI,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_val = func(new_x.detach().squeeze().numpy())
            train_x.append(new_x.squeeze(0))
            train_y.append(new_val)

        train_y = np.array(train_y)
        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()