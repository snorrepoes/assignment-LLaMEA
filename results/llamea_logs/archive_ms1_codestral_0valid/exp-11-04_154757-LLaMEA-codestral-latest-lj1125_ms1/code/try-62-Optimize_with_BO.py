import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from scipy.stats import qmc

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.f_opt = np.inf
        self.x_opt = None
        self.evaluated_points = []
        self.evaluated_values = []

    def __call__(self, func):
        # Evaluate initial points
        for x in self.initial_points:
            f = func(x)
            self.evaluated_values.append(f)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x.copy()

        # Convert to torch tensors
        train_x = torch.tensor(np.array(self.initial_points), dtype=torch.float64)
        train_y = torch.tensor(np.array(self.evaluated_values), dtype=torch.float64).unsqueeze(-1)

        # Log-transform and standardize
        train_y_log = torch.log(torch.abs(train_y) + 1e-8)
        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.float64)
        train_x_normalized = normalize(train_x, bounds)

        remaining_budget = self.budget - len(self.initial_points)

        for _ in range(remaining_budget):
            # Create GP model
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            gp = SingleTaskGP(train_x_normalized, train_y_log, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll.to(train_x_normalized)
            fit_gpytorch_model(mll)  # This is the corrected line

            # Acquisition function
            acqf = qLogNoisyExpectedImprovement(
                model=gp,
                best_f=train_y_log.min(),
                X_baseline=train_x_normalized,
                prune_baseline=True
            )

            # Optimize
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )

            # Evaluate new point
            candidate = candidates.detach().squeeze().numpy()
            candidate_denormalized = bounds[0] + candidate * (bounds[1] - bounds[0])
            f = func(candidate_denormalized)

            # Update best
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = candidate_denormalized.copy()

            # Update training data
            train_x = torch.cat([train_x, torch.tensor(candidate_denormalized, dtype=torch.float64).unsqueeze(0)])
            train_y = torch.cat([train_y, torch.tensor([[f]], dtype=torch.float64)])
            train_y_log = torch.log(torch.abs(train_y) + 1e-8)
            train_x_normalized = normalize(train_x, bounds)

        return self.f_opt, self.x_opt