import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel

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
            
            # Using Standardize transform to handle plateau/well dynamics
            model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Maximize negative f (minimization) = Maximize Expected Improvement
            best_f = Y.min()
            acq = LogExpectedImprovement(model=model, best_f=-best_f, maximize=True)
            
            # Optimize acquisition with local restarts to prevent plateau stagnation
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=20,
                raw_samples=512,
            )
            
            next_x = new_x.detach().numpy()[0]
            train_x.append(torch.tensor(next_x, dtype=torch.double))
            train_y.append(func(next_x))

        best_idx = np.argmin(train_y)
        return float(train_y[best_idx]), train_x[best_idx].numpy()