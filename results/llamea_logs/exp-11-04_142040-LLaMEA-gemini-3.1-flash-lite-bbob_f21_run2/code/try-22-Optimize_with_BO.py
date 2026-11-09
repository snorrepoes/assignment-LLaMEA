import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
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
        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.double)
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in train_x], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        
        # State variables for local trust region
        length = 0.8
        min_length = 0.01
        max_length = 1.6
        success_counter = 0
        failure_counter = 0
        success_tol = 3
        failure_tol = 5

        while num_calls < self.budget:
            train_x_norm = normalize(train_x, bounds)
            
            # Fit GP
            likelihood = GaussianLikelihood()
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x_norm, train_y, likelihood=likelihood, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define search region around best point
            best_idx = torch.argmin(train_y)
            center = train_x_norm[best_idx]
            tr_lb = torch.clamp(center - length / 2, 0.0, 1.0)
            tr_ub = torch.clamp(center + length / 2, 0.0, 1.0)

            qLEI = qLogExpectedImprovement(model, best_f=-torch.min(train_y))
            candidate_norm, _ = optimize_acqf(
                acq_function=qLEI,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=15,
                raw_samples=512,
            )

            new_x = unnormalize(candidate_norm, bounds).detach().squeeze(0).numpy()
            new_y = func(new_x)
            
            # Update state
            if new_y < torch.min(train_y).item():
                success_counter += 1
                failure_counter = 0
            else:
                success_counter = 0
                failure_counter += 1
            
            if success_counter >= success_tol:
                length = min(2.0 * length, max_length)
                success_counter = 0
            elif failure_counter >= failure_tol:
                length = length / 2.0
                failure_counter = 0
            
            train_x = torch.cat([train_x, torch.tensor([new_x], dtype=torch.double)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1
            
            if length < min_length:
                length = 0.8 # Reset TR

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()