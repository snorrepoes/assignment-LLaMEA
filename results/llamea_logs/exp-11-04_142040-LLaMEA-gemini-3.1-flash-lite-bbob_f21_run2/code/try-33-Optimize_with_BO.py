import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.utils.transforms import normalize, unnormalize

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
        train_y = torch.tensor([func(x) for x in train_x], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        if num_calls > self.budget:
            raise ValueError("Initial points exceed budget.")

        # TuRBO-1 parameters
        length = 0.8
        min_length = 0.01
        max_length = 1.6
        success_counter = 0
        failure_counter = 0
        success_tol = 3
        failure_tol = 5

        while num_calls < self.budget:
            train_x_norm = normalize(train_x, bounds)
            
            # Use GP with ARD Matern 2.5
            likelihood = GaussianLikelihood()
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x_norm, -train_y, likelihood=likelihood, covar_module=covar_module) # Maximize negative
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define trust region around current best
            best_idx = torch.argmax(-train_y)
            center = train_x_norm[best_idx]
            tr_lb = torch.clamp(center - length / 2, 0, 1)
            tr_ub = torch.clamp(center + length / 2, 0, 1)

            qNEI = qNoisyExpectedImprovement(model, X_baseline=train_x_norm)
            
            candidate_norm, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=10,
                raw_samples=256,
            )

            new_x = unnormalize(candidate_norm, bounds).detach().squeeze(0).numpy()
            new_y = func(new_x)
            
            # Update trust region logic
            if new_y < torch.min(train_y):
                success_counter += 1
                failure_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
                
            if success_counter >= success_tol:
                length = min(2.0 * length, max_length)
                success_counter = 0
            elif failure_counter >= failure_tol:
                length = length / 2
                failure_counter = 0

            train_x = torch.cat([train_x, torch.tensor([new_x], dtype=torch.double)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1
            if length < min_length: break

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()