import torch
import numpy as np
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import LogStandardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
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
            
            # Using LogStandardize to handle high dynamic range and flatten the plateau influence
            model = SingleTaskGP(
                X, Y, 
                outcome_transform=LogStandardize(),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Adaptive Trust Region (TuRBO-1 style)
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Length scale adaptation based on successful iterations
            tr_length = 2.0 * (0.95 ** (len(train_x) / 10.0))
            tr_lb = torch.clamp(best_x - tr_length, lb, ub)
            tr_ub = torch.clamp(best_x + tr_length, lb, ub)

            # Use qLogEI for better numerical stability in deep wells
            qLogEI = qLogExpectedImprovement(model, best_f=Y.min())
            
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

        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()