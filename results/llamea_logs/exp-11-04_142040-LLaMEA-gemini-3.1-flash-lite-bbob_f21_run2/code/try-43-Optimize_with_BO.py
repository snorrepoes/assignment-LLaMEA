import torch
import numpy as np
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from botorch.acquisition.monte_carlo import qSimpleRegret

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
        length = 1.6
        min_length = 0.05
        max_length = 1.6
        
        while num_calls < self.budget:
            train_y_min = train_y.min()
            
            # Use a Matern 5/2 kernel with ARD
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1),
                covar_module=covar_module
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Trust Region centered on best observed point
            best_x = train_x[train_y.argmin()]
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            # Thompson Sampling for robust exploration
            acq = qSimpleRegret(model)
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=tr_bounds,
                q=1,
                num_restarts=15,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200}
            )

            new_x = candidate.detach().flatten().numpy()
            new_y = func(new_x)
            
            # Adaptive Trust Region logic
            if new_y < train_y_min:
                length = min(length * 1.1, max_length)
            else:
                length = max(length * 0.75, min_length)
            
            train_x = torch.cat([train_x, candidate.detach()])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1
            
            # Reset if stuck
            if length <= min_length * 1.1:
                length = max_length

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()