import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        bounds = torch.stack([lb, ub])
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        if num_calls > self.budget:
            raise ValueError("Initial points exceed budget.")

        # TuRBO-like state tracking
        length = 0.8
        success_count = 0
        failure_count = 0
        
        while num_calls < self.budget:
            train_y_std = train_y.std()
            outcome_transform = Standardize(m=1) if train_y_std > 1e-6 else None
            
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                train_x, train_y, 
                covar_module=covar_module,
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=outcome_transform
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq = qNoisyExpectedImprovement(
                model=model, 
                X_baseline=train_x, 
                prune_baseline=True
            )

            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )

            new_x = candidate.detach().flatten().numpy()
            new_y = func(new_x)
            
            train_x = torch.cat([train_x, candidate.detach()])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()